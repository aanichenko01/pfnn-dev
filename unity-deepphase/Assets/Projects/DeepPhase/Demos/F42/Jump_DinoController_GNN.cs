using UnityEngine;
using AI4Animation;
using UltimateIK;
using System.Collections.Generic;
using Unity.Barracuda;
using System;

namespace DeepPhase
{
    public class Jump_DinoController_GNN : AnimationController
    {

        public ONNXNetwork NeuralNetwork;

        public Camera Camera;

        public bool DrawGUI = true;
        public bool DrawDebug = true;

        public int Channels = 10;
        public float MoveSpeed = 1.5f;
        public float SprintSpeed = 4f;

        public float RootDampening = 1f;
        [Range(0f, 1f)] public float PositionBias = 1f;
        [Range(0f, 1f)] public float DirectionBias = 1f;
        [Range(0f, 1f)] public float VelocityBias = 1f;
        [Range(0f, 1f)] public float ControlWeight = 1f / 3f;
        [Range(0f, 1f)] public float CorrectionWeight = 1f / 3f;
        [Range(0f, 1f)] public float PhaseStability = 0.5f;

        public bool Postprocessing = true;
        public float ContactPower = 3f;
        public float ContactThreshold = 2f / 3f;

        public bool[] ActivePhases = new bool[0];

        private InputSystem Controller;

        private TimeSeries TimeSeries;
        private RootModule.Series RootSeries;
        private StyleModule.Series StyleSeries;
        private ContactModule.Series ContactSeries;
        private DeepPhaseModule.Series PhaseSeries;

        private IK LeftFootIK;
        private IK RightFootIK;

        private float[] PreviousPositionsMagnitude;
		private float TotalMagnitudeDiff;
		private float TimeElapsed;
		private float TimeInterval = 1f;

		private float[] PreviousY;
		private float[] PreviousZ;

		private float TotalYDiff;
		private float TotalZDiff;

		private int Count;

        public void RetrieveContacts(List<Vector4> contacts) {
            void Accumulate(string name, int index) {
                Vector3 vector = Actor.GetBonePosition(name);
                float contact = ContactSeries.Values[TimeSeries.Pivot][index];
                contacts.Add(new Vector4(vector.x, vector.y, vector.z, contact));
            }
            Accumulate("AnzB:LeftFootIndex4", 0);
            Accumulate("AnzB:RightFootIndex4", 1);
        }

        protected override void Setup() {	
            Controller = new InputSystem(1);

            InputSystem.Logic idle = Controller.AddLogic("Idle", () => Controller.QueryLeftJoystickVector().magnitude < 0.1f && Controller.QueryRightJoystickVector().magnitude < 0.1f);
            InputSystem.Logic move = Controller.AddLogic("Move", () => !idle.Query());
            InputSystem.Logic speed = Controller.AddLogic("Speed", () => true);
            InputSystem.Logic jump = Controller.AddLogic("Jump", () => Controller.QueryJumpKeyboard()); //for now to test if working if moving jump is assumed

            TimeSeries = new TimeSeries(6, 6, 1f, 1f, 10);
            RootSeries = new RootModule.Series(TimeSeries, transform);
            StyleSeries = new StyleModule.Series(TimeSeries, new string[] { "Idle", "Move", "Speed", "Jump" }, new float[] { 1f, 0f, 0f, 0f });
            ContactSeries = new ContactModule.Series(TimeSeries, "Left Foot", "Right Foot");
            PhaseSeries = new DeepPhaseModule.Series(TimeSeries, Channels);

            LeftFootIK = IK.Create(Actor.FindTransform("AnzB:LeftLeg"), Actor.GetBoneTransforms("AnzB:LeftFootIndex4"));
            RightFootIK = IK.Create(Actor.FindTransform("AnzB:RightLeg"), Actor.GetBoneTransforms("AnzB:RightFootIndex4"));

            RootSeries.DrawGUI = DrawGUI;
            StyleSeries.DrawGUI = DrawGUI;
            ContactSeries.DrawGUI = DrawGUI;
            PhaseSeries.DrawGUI = DrawGUI;
            RootSeries.DrawScene = DrawDebug;
            StyleSeries.DrawScene = DrawDebug;
            ContactSeries.DrawScene = DrawDebug;
            PhaseSeries.DrawScene = DrawDebug;

            ActivePhases = new bool[PhaseSeries.Channels];
            ActivePhases.SetAll(true);

            NeuralNetwork.CreateSession();

            // Metric variables
            PreviousPositionsMagnitude = new float[Actor.Bones.Length];
			
			PreviousY = new float[Actor.Bones.Length];
			PreviousZ = new float[Actor.Bones.Length];

			TotalMagnitudeDiff = 0f;
			TimeElapsed = 0f;
			Count = 0;

            // Setup
            for (int i = 0; i < Actor.Bones.Length; i++)
			{
                Vector3 zyPos = Actor.Bones[i].GetPosition();
                zyPos.x = 0f;
				PreviousPositionsMagnitude[i] = zyPos.magnitude;
				PreviousY[i] = Actor.Bones[i].GetPosition().y;
				PreviousZ[i] = Actor.Bones[i].GetPosition().z;
			}

        }

        protected override void Destroy()
        {
            NeuralNetwork.CloseSession();
        }

        protected override void Control()
        {
            UserControl();
            Feed();
            NeuralNetwork.RunSession();
            Read();

            // Metrics
            TimeElapsed += Time.deltaTime;
            if (TimeElapsed >= TimeInterval) {
				Count += 1;
				for (int i = 0; i < Actor.Bones.Length; i++)
				{
					Vector3 zyPos = Actor.Bones[i].GetPosition();
					zyPos.x = 0f;
					float currentMagnitude = zyPos.magnitude;

					TotalMagnitudeDiff += Math.Abs(currentMagnitude - PreviousPositionsMagnitude[i]);
					TotalYDiff += Math.Abs(Actor.Bones[i].GetPosition().y - PreviousY[i]);
					TotalZDiff += Math.Abs(Actor.Bones[i].GetPosition().z - PreviousZ[i]);
					
					PreviousPositionsMagnitude[i] = currentMagnitude;
					PreviousY[i] = Actor.Bones[i].GetPosition().y;
					PreviousZ[i] = Actor.Bones[i].GetPosition().z;
				}

				float avgMagnitudeDiff = TotalMagnitudeDiff / (Actor.Bones.Length * TimeInterval * Count);
				float avgYDiff = TotalYDiff / ( Actor.Bones.Length* TimeInterval * Count);
				float avgZDiff = TotalZDiff / ( Actor.Bones.Length* TimeInterval * Count);
            	Debug.Log($"[{Count}]Average change in magnitude per second: {avgMagnitudeDiff}");
				Debug.Log($"[{Count}]Average change in Y per second: {avgYDiff}");
				Debug.Log($"[{Count}]Average change in Z per second: {avgZDiff}");
				// TotalMagnitudeDiff = 0f;
                // TotalYDiff = 0f;
                // TotalZDiff = 0f;
                TimeElapsed = 0f;
			}

        }

        private void UserControl()
        {
            //Update User Controller Inputs
            Controller.Update();

            //Locomotion
            Vector3 move = Controller.QueryLeftJoystickVector();
            move = move.magnitude < 0.25f ? Vector3.zero : move;
            Vector3 face = Controller.QueryRightJoystickVector().ZeroY();
            face = face.magnitude < 0.25f ? move : face;

            //Amplify Factors
            move = Quaternion.LookRotation(Vector3.ProjectOnPlane(transform.position - Camera.transform.position, Vector3.up).normalized, Vector3.up) * move;
            move *= MoveSpeed;

            //Trajectory
            RootSeries.Control(move, face, ControlWeight, PositionBias, DirectionBias, VelocityBias);

            //Action Values
            float[] actions = Controller.PoolLogics(StyleSeries.Styles);
            actions[StyleSeries.Styles.FindIndex("Speed")] *= move.magnitude;
            StyleSeries.Control(actions, ControlWeight);
        }

        private void Feed()
        {
            //Get Root
            Matrix4x4 root = Actor.GetRoot().GetWorldMatrix();

            //Input Timeseries
            for (int i = 0; i < TimeSeries.KeyCount; i++)
            {
                int index = TimeSeries.GetKey(i).Index;
                NeuralNetwork.FeedXZ(RootSeries.GetPosition(index).PositionTo(root));
                NeuralNetwork.FeedXZ(RootSeries.GetDirection(index).DirectionTo(root));
                NeuralNetwork.FeedXZ(RootSeries.Velocities[index].DirectionTo(root));
                NeuralNetwork.Feed(StyleSeries.Values[index]);
            }

            //Input Character
            for (int i = 0; i < Actor.Bones.Length; i++)
            {
                NeuralNetwork.Feed(Actor.Bones[i].GetTransform().position.PositionTo(root));
                NeuralNetwork.Feed(Actor.Bones[i].GetTransform().forward.DirectionTo(root));
                NeuralNetwork.Feed(Actor.Bones[i].GetTransform().up.DirectionTo(root));
                NeuralNetwork.Feed(Actor.Bones[i].GetVelocity().DirectionTo(root));
            }

            //Input Gating Features
            NeuralNetwork.Feed(PhaseSeries.GetAlignment());
        }

        private void Read()
        {
            //Update Past States
            ContactSeries.Increment(0, TimeSeries.Pivot);
            PhaseSeries.Increment(0, TimeSeries.Pivot);

            //Update Root State
            Vector3 offset = NeuralNetwork.ReadVector3();
            float update = StyleSeries.GetValue(TimeSeries.Pivot, "Speed");
            update = Mathf.Pow(Mathf.Clamp(update, 0f, 1f), RootDampening);
            offset = Vector3.Lerp(Vector3.zero, offset, update);

            Matrix4x4 root = Actor.GetRoot().GetWorldMatrix() * Matrix4x4.TRS(new Vector3(offset.x, 0f, offset.z), Quaternion.AngleAxis(offset.y, Vector3.up), Vector3.one);
            RootSeries.Transformations[TimeSeries.Pivot] = root;
            RootSeries.Velocities[TimeSeries.Pivot] = NeuralNetwork.ReadXZ().DirectionFrom(root);
            for (int j = 0; j < StyleSeries.Styles.Length; j++)
            {
                StyleSeries.Values[TimeSeries.Pivot][j] = Mathf.Lerp(
                    StyleSeries.Values[TimeSeries.Pivot][j],
                    NeuralNetwork.Read(),
                    CorrectionWeight
                );
            }

            //Read Future States
            for (int i = TimeSeries.PivotKey + 1; i < TimeSeries.KeyCount; i++)
            {
                int index = TimeSeries.GetKey(i).Index;

                Matrix4x4 m = Matrix4x4.TRS(NeuralNetwork.ReadXZ().PositionFrom(root), Quaternion.LookRotation(NeuralNetwork.ReadXZ().DirectionFrom(root).normalized, Vector3.up), Vector3.one);
                RootSeries.Transformations[index] = Utility.Interpolate(RootSeries.Transformations[index], m,
                    CorrectionWeight,
                    CorrectionWeight
                );
                RootSeries.Velocities[index] = Vector3.Lerp(RootSeries.Velocities[index], NeuralNetwork.ReadXZ().DirectionFrom(root),
                    CorrectionWeight
                );

                for (int j = 0; j < StyleSeries.Styles.Length; j++)
                {
                    StyleSeries.Values[index][j] = Mathf.Lerp(StyleSeries.Values[index][j], NeuralNetwork.Read(), CorrectionWeight);
                }
            }

            //Read Posture
            Vector3[] positions = new Vector3[Actor.Bones.Length];
            Vector3[] forwards = new Vector3[Actor.Bones.Length];
            Vector3[] upwards = new Vector3[Actor.Bones.Length];
            Vector3[] velocities = new Vector3[Actor.Bones.Length];
            for (int i = 0; i < Actor.Bones.Length; i++)
            {
                Vector3 position = NeuralNetwork.ReadVector3().PositionFrom(root);
                Vector3 forward = forward = NeuralNetwork.ReadVector3().normalized.DirectionFrom(root);
                Vector3 upward = NeuralNetwork.ReadVector3().normalized.DirectionFrom(root);
                Vector3 velocity = NeuralNetwork.ReadVector3().DirectionFrom(root);
                if (!Actor.Bones[i].GetName().Contains("Tail"))
                {
                    if (Controller.QueryLogic("Idle") && update < 1f)
                    {
                        float weight = update.Normalize(0f, 1f, 0.25f, 1f);
                        position = Vector3.Lerp(Actor.Bones[i].GetPosition(), position, weight);
                        forward = Vector3.Slerp(Actor.Bones[i].GetTransform().forward, forward, weight);
                        upward = Vector3.Slerp(Actor.Bones[i].GetTransform().up, upward, weight);
                        velocity = Vector3.Lerp(Actor.Bones[i].GetVelocity(), velocity, weight);
                    }
                }
                if (Actor.Bones[i].GetName().Contains("Head"))
                {
                    forward = Vector3.Slerp(Actor.Bones[i].GetTransform().forward, forward, 0.5f);
                    upward = Vector3.Slerp(Actor.Bones[i].GetTransform().up, upward, 0.5f);
                }
                velocities[i] = velocity;
                positions[i] = Vector3.Lerp(Actor.Bones[i].GetTransform().position + velocity / Framerate, position, 0.5f);
                forwards[i] = forward;
                upwards[i] = upward;

            }


            //Update Contacts
            float[] contacts = NeuralNetwork.Read(ContactSeries.Bones.Length, 0f, 1f);
            for (int i = 0; i < ContactSeries.Bones.Length; i++)
            {
                ContactSeries.Values[TimeSeries.Pivot][i] = contacts[i].SmoothStep(ContactPower, ContactThreshold);
            }

            //Update Phases
            PhaseSeries.UpdateAlignment(NeuralNetwork.Read((1 + PhaseSeries.FutureKeys) * PhaseSeries.Channels * 4), PhaseStability, 1f / Framerate);

            //Interpolate Timeseries
            RootSeries.Interpolate(TimeSeries.Pivot, TimeSeries.Samples.Length);
            StyleSeries.Interpolate(TimeSeries.Pivot, TimeSeries.Samples.Length);
            PhaseSeries.Interpolate(TimeSeries.Pivot, TimeSeries.Samples.Length);

            //Assign Posture
            transform.position = RootSeries.GetPosition(TimeSeries.Pivot);
            transform.rotation = RootSeries.GetRotation(TimeSeries.Pivot);
            for (int i = 0; i < Actor.Bones.Length; i++)
            {
                Actor.Bones[i].SetVelocity(velocities[i]);
                Actor.Bones[i].SetPosition(positions[i]);
                Actor.Bones[i].SetRotation(Quaternion.LookRotation(forwards[i], upwards[i]));
            }

            //Correct Twist
            Actor.RestoreAlignment();

            for (int i = 0; i < PhaseSeries.Channels; i++)
            {
                if (!ActivePhases[i])
                {
                    for (int j = 0; j < PhaseSeries.Samples.Length; j++)
                    {
                        PhaseSeries.Amplitudes[j][i] = 0f;
                    }
                }
            }

            //Process Contact States
            // ProcessFootIK(LeftHandIK, ContactSeries.Values[TimeSeries.Pivot][0]);
            // ProcessFootIK(RightHandIK, ContactSeries.Values[TimeSeries.Pivot][1]);
            // For now comment out footIK might be messing with it?
            // ProcessFootIK(LeftFootIK, ContactSeries.Values[TimeSeries.Pivot][0]);
            // ProcessFootIK(RightFootIK, ContactSeries.Values[TimeSeries.Pivot][1]);

            PhaseSeries.AddGatingHistory(NeuralNetwork.GetOutput("W").AsFloats());
        }

        private void ProcessFootIK(IK ik, float contact)
        {
            if (!Postprocessing)
            {
                return;
            }
            ik.Activation = UltimateIK.ACTIVATION.Linear;
            for (int i = 0; i < ik.Objectives.Length; i++)
            {
                ik.Objectives[i].SetTarget(Vector3.Lerp(ik.Objectives[i].TargetPosition, ik.Joints[ik.Objectives[i].Joint].Transform.position, 1f - contact));
                ik.Objectives[i].SetTarget(ik.Joints[ik.Objectives[i].Joint].Transform.rotation);
            }
            ik.Iterations = 25;
            ik.Solve();
        }

        protected override void OnGUIDerived()
        {
            RootSeries.DrawGUI = DrawGUI;
            // StyleSeries.DrawGUI = DrawGUI;
            // ContactSeries.DrawGUI = DrawGUI;
            // PhaseSeries.DrawGUI = DrawGUI;
            RootSeries.GUI();
            // StyleSeries.GUI();
            // ContactSeries.GUI();
            // PhaseSeries.GUI();
        }

        protected override void OnRenderObjectDerived()
        {
            RootSeries.DrawScene = DrawDebug;
            StyleSeries.DrawScene = DrawDebug;
            ContactSeries.DrawScene = DrawDebug;
            PhaseSeries.DrawScene = DrawDebug;
            RootSeries.Draw();
            // StyleSeries.Draw();
            // ContactSeries.Draw();
            // PhaseSeries.Draw();
        }

    }
}