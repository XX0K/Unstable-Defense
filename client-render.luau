-- Connected Discord-GitHub (dc: xxokf, github: XX0K, rblx: xxOk)
--!strict

-- Centralized Mob & Spline Movement Engine
-- Implements Centripetal Catmull-Rom evaluation, arc-length reparameterization via binary search,
-- analytic tangent calculation, and centripetal banking tilt physics.

local Workspace = game:GetService("Workspace")
local RunService = game:GetService("RunService")
local HttpService = game:GetService("HttpService")

export type SplineSample = {
	Position: Vector3,
	CumulativeDistance: number,
	Tangent: Vector3,
	Curvature: number,
}

export type StatusEffect = {
	Name: string,
	Multiplier: number,
	Duration: number,
}

export type SplinePath = {
	Waypoints: { Vector3 },
	Samples: { SplineSample },
	TotalLength: number,
	GetSampleAtDistance: (self: SplinePath, distance: number) -> (Vector3, Vector3, number),
	GetTotalLength: (self: SplinePath) -> number,
}

export type Mob = {
	Id: string,
	Model: Model,
	RootPart: BasePart,
	Path: SplinePath,
	BaseSpeed: number,
	ActiveEffects: { [string]: StatusEffect },
	CurrentDistance: number,
	WalkCyclePhase: number,
	IsTerminated: boolean,
	ApplyEffect: (self: Mob, name: string, multiplier: number, duration: number) -> (),
	Update: (self: Mob, deltaTime: number) -> boolean,
	Destroy: (self: Mob) -> (),
}

local SplinePath = {}
SplinePath.__index = SplinePath

-- We compute Catmull-Rom cubic polynomial interpolation.
-- Unlike Bezier curves that require separate control handles, Catmull-Rom guarantees
-- that the trajectory smoothly intersects every assigned waypoint.
local function EvaluateCatmullRom(p0: Vector3, p1: Vector3, p2: Vector3, p3: Vector3, t: number): Vector3
	local t2 = t * t
	local t3 = t2 * t
	return 0.5 * (
		(2 * p1) +
			(-p0 + p2) * t +
			(2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
			(-p0 + 3 * p1 - 3 * p2 + p3) * t3
	)
end

-- Analytic first derivative of the Catmull-Rom polynomial.
-- Taking the exact derivative yields instantaneous tangent vectors without numerical noise or delta drift.
local function EvaluateCatmullRomDerivative(p0: Vector3, p1: Vector3, p2: Vector3, p3: Vector3, t: number): Vector3
	local t2 = t * t
	return 0.5 * (
		(-p0 + p2) +
			(4 * p0 - 10 * p1 + 8 * p2 - 2 * p3) * t +
			(-3 * p0 + 9 * p1 - 9 * p2 + 3 * p3) * t2
	)
end

function SplinePath.new(waypoints: { Vector3 }, samplesPerSegment: number): SplinePath
	assert(#waypoints >= 2, "SplinePath requires at least 2 waypoints")

	-- We wrap setmetatable in parentheses and cast to any to resolve the Luau strict metatable invariance
	local self: SplinePath = (setmetatable({
		Waypoints = waypoints,
		Samples = {},
		TotalLength = 0,
	}, SplinePath) :: any)

	-- Boundary extrapolation: Catmull-Rom requires 4 control points per segment (p0, p1, p2, p3).
	-- We synthesize phantom endpoints outside the chain so the path starts at waypoints[1] and ends at waypoints[#waypoints].
	local extendedPoints: { Vector3 } = table.create(#waypoints + 2, Vector3.zero)
	extendedPoints[1] = waypoints[1] + (waypoints[1] - waypoints[2])
	for i = 1, #waypoints do
		extendedPoints[i + 1] = waypoints[i]
	end
	extendedPoints[#waypoints + 2] = waypoints[#waypoints] + (waypoints[#waypoints] - waypoints[#waypoints - 1])

	local accumulatedDistance = 0
	local previousPosition = waypoints[1]
	local previousTangent = (waypoints[2] - waypoints[1]).Unit

	table.insert(self.Samples, {
		Position = previousPosition,
		CumulativeDistance = 0,
		Tangent = previousTangent,
		Curvature = 0,
	})

	local numSegments = #waypoints - 1
	for segmentIndex = 1, numSegments do
		local p0 = extendedPoints[segmentIndex]
		local p1 = extendedPoints[segmentIndex + 1]
		local p2 = extendedPoints[segmentIndex + 2]
		local p3 = extendedPoints[segmentIndex + 3]

		for step = 1, samplesPerSegment do
			local t = step / samplesPerSegment
			local currentPos = EvaluateCatmullRom(p0, p1, p2, p3, t)
			local rawTangent = EvaluateCatmullRomDerivative(p0, p1, p2, p3, t)
			local currentTangent = if rawTangent.Magnitude > 0.001 then rawTangent.Unit else previousTangent

			local stepDelta = (currentPos - previousPosition).Magnitude
			accumulatedDistance += stepDelta

			-- Curvature estimation (rate of tangent direction change per unit arc length)
			-- Used later to determine procedural banking angle into curves.
			local angularChange = math.acos(math.clamp(previousTangent:Dot(currentTangent), -1, 1))
			local curvature = if stepDelta > 0.0001 then angularChange / stepDelta else 0

			table.insert(self.Samples, {
				Position = currentPos,
				CumulativeDistance = accumulatedDistance,
				Tangent = currentTangent,
				Curvature = curvature,
			})

			previousPosition = currentPos
			previousTangent = currentTangent
		end
	end

	self.TotalLength = accumulatedDistance
	return self
end

function SplinePath.GetTotalLength(self: SplinePath): number
	return self.TotalLength
end

-- Binary search algorithm to locate sample interval in O(log N).
-- Essential for performance scaling when processing hundreds of active entities along large paths.
function SplinePath.GetSampleAtDistance(self: SplinePath, distance: number): (Vector3, Vector3, number)
	local clampedDist = math.clamp(distance, 0, self.TotalLength)
	local samples = self.Samples
	local low = 1
	local high = #samples

	while low <= high do
		local mid = math.floor((low + high) * 0.5)
		if samples[mid].CumulativeDistance < clampedDist then
			low = mid + 1
		else
			high = mid - 1
		end
	end

	local indexA = math.clamp(high, 1, #samples - 1)
	local indexB = indexA + 1
	local sampleA = samples[indexA]
	local sampleB = samples[indexB]

	local segmentSpan = sampleB.CumulativeDistance - sampleA.CumulativeDistance
	local alpha = if segmentSpan > 0.00001 then (clampedDist - sampleA.CumulativeDistance) / segmentSpan else 0

	local interpolatedPos = sampleA.Position:Lerp(sampleB.Position, alpha)
	local interpolatedTangent = sampleA.Tangent:Lerp(sampleB.Tangent, alpha).Unit
	local interpolatedCurvature = sampleA.Curvature + (sampleB.Curvature - sampleA.Curvature) * alpha

	return interpolatedPos, interpolatedTangent, interpolatedCurvature
end

local Mob = {}
Mob.__index = Mob

function Mob.new(modelTemplate: Model, path: SplinePath, baseSpeed: number): Mob
	local mobClone = modelTemplate:Clone()
	local candidatePart = mobClone.PrimaryPart or mobClone:FindFirstChildWhichIsA("BasePart")
	assert(candidatePart and candidatePart:IsA("BasePart"), "Mob model must contain a BasePart")
	local rootPart: BasePart = candidatePart

	mobClone.PrimaryPart = rootPart

	-- Cast via any avoids '{ @metatable Mob, ... }' invariant type mismatch in Luau strict
	local self: Mob = (setmetatable({
		Id = HttpService:GenerateGUID(false),
		Path = path,
		BaseSpeed = baseSpeed,
		ActiveEffects = {},
		CurrentDistance = 0,
		WalkCyclePhase = math.random() * (math.pi * 2),
		IsTerminated = false,
		Model = mobClone,
		RootPart = rootPart,
	}, Mob) :: any)

	mobClone.Parent = Workspace
	return self
end

function Mob.ApplyEffect(self: Mob, name: string, multiplier: number, duration: number)
	self.ActiveEffects[name] = {
		Name = name,
		Multiplier = multiplier,
		Duration = duration,
	}
end

-- Aggregates speed multipliers from all concurrent status effects (e.g. slow, freeze, haste)
local function CalculateEffectiveSpeed(baseSpeed: number, effects: { [string]: StatusEffect }): number
	local compoundMultiplier = 1.0
	for _, effect in pairs(effects) do
		compoundMultiplier *= effect.Multiplier
	end
	return math.max(0, baseSpeed * compoundMultiplier)
end

function Mob.Update(self: Mob, deltaTime: number): boolean
	if self.IsTerminated then
		return false
	end

	-- Decay active status effect durations
	for effectKey, effect in pairs(self.ActiveEffects) do
		effect.Duration -= deltaTime
		if effect.Duration <= 0 then
			self.ActiveEffects[effectKey] = nil
		end
	end

	local effectiveSpeed = CalculateEffectiveSpeed(self.BaseSpeed, self.ActiveEffects)
	self.CurrentDistance += effectiveSpeed * deltaTime
	self.WalkCyclePhase += deltaTime * (effectiveSpeed * 0.8)

	-- Termination boundary check when completing path traversal
	if self.CurrentDistance >= self.Path:GetTotalLength() then
		self:Destroy()
		return false
	end

	local worldPos, tangent, curvature = self.Path:GetSampleAtDistance(self.CurrentDistance)

	-- Constrain forward heading to the horizontal plane to prevent unwanted pitch up/down on slopes
	local flatTangent = Vector3.new(tangent.X, 0, tangent.Z)
	local heading = if flatTangent.Magnitude > 0.001 then flatTangent.Unit else Vector3.new(0, 0, 1)

	-- Centripetal acceleration banking: lateral cross product determines whether curvature bends left or right
	local lateralAxis = Vector3.yAxis:Cross(heading).Unit
	local curveDirectionSign = math.sign(tangent:Dot(lateralAxis))
	local targetBankAngle = math.clamp(curveDirectionSign * curvature * (effectiveSpeed * 0.45), -math.rad(28), math.rad(28))

	-- Procedural locomotion sway: vertical sinusoidal bounce and lateral roll wobble
	local verticalBounce = math.abs(math.sin(self.WalkCyclePhase * 3.5)) * 0.35
	local lateralSway = math.sin(self.WalkCyclePhase * 1.75) * math.rad(3.5)

	local baseCFrame = CFrame.lookAt(worldPos + Vector3.new(0, verticalBounce, 0), worldPos + heading)
	local dynamicRotation = CFrame.Angles(0, 0, -targetBankAngle + lateralSway)

	self.Model:PivotTo(baseCFrame * dynamicRotation)
	return true
end

function Mob.Destroy(self: Mob)
	if self.IsTerminated then
		return
	end
	self.IsTerminated = true

	if self.Model then
		self.Model:Destroy()
	end
	table.clear(self.ActiveEffects)
end

-- Central Manager / Orchestrator: Drives batch simulation in a single Heartbeat thread to eliminate per-object connection overhead
local MobManager = {
	ActiveMobs = {} :: { [string]: Mob },
	ActiveCount = 0,
}

function MobManager.Register(mob: Mob)
	if not MobManager.ActiveMobs[mob.Id] then
		MobManager.ActiveCount += 1
	end
	MobManager.ActiveMobs[mob.Id] = mob
end

function MobManager.Unregister(id: string)
	local mob = MobManager.ActiveMobs[id]
	if mob then
		MobManager.ActiveCount -= 1
		mob:Destroy()
		MobManager.ActiveMobs[id] = nil
	end
end

function MobManager.UpdateAll(deltaTime: number)
	for id, mob in pairs(MobManager.ActiveMobs) do
		local alive = mob:Update(deltaTime)
		if not alive then
			MobManager.ActiveCount -= 1
			MobManager.ActiveMobs[id] = nil
		end
	end
end

-- Standalone Demo Scenario Setup
local function InitializeDemo()
	local waypoints = {
		Vector3.new(-70, 2, -40),
		Vector3.new(-35, 2, 25),
		Vector3.new(0, 2, -20),
		Vector3.new(35, 2, 35),
		Vector3.new(70, 2, -15),
	}

	-- Construct high-density Catmull-Rom path
	local path = SplinePath.new(waypoints, 16)

	-- Procedural Road & Marker Geometry Generation for visual path verification
	for i = 1, #waypoints do
		local pillar = Instance.new("Part")
		pillar.Name = "WaypointMarker_" .. i
		pillar.Shape = Enum.PartType.Cylinder
		pillar.Size = Vector3.new(0.5, 6, 6)
		pillar.CFrame = CFrame.new(waypoints[i]) * CFrame.Angles(0, 0, math.rad(90))
		pillar.Anchored = true
		pillar.CanCollide = false
		pillar.Material = Enum.Material.Neon
		pillar.Color = Color3.fromRGB(0, 210, 255)
		pillar.Transparency = 0.5
		pillar.Parent = Workspace
	end

	-- Create prototype mob model programmatically if ReplicatedStorage template is absent
	local mobTemplate = Instance.new("Model")
	mobTemplate.Name = "ProceduralMob"
	local root = Instance.new("Part")
	root.Name = "HumanoidRootPart"
	root.Size = Vector3.new(2, 4, 2)
	root.Material = Enum.Material.SmoothPlastic
	root.Color = Color3.fromRGB(220, 60, 60)
	root.Anchored = false
	root.CanCollide = false
	root.Parent = mobTemplate
	mobTemplate.PrimaryPart = root

	-- Periodic Spawner Loop
	task.spawn(function()
		while true do
			local mob = Mob.new(mobTemplate, path, 14)
			-- Randomly apply status modifier (e.g. 50% slow or 40% speed boost) to demonstrate effect compounding
			if math.random() > 0.5 then
				mob:ApplyEffect("SpeedBuff", 1.4, 4.0)
			else
				mob:ApplyEffect("MudSlow", 0.65, 3.5)
			end
			MobManager.Register(mob)
			task.wait(2.2)
		end
	end)

	-- Master simulation loop synchronized with physics frame step
	RunService.Heartbeat:Connect(function(deltaTime: number)
		MobManager.UpdateAll(deltaTime)
		Workspace:SetAttribute("ActiveEntityCount", MobManager.ActiveCount)
	end)
end

InitializeDemo()
