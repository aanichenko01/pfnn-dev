﻿using UnityEngine;
using System;
using System.Collections.Generic;
using DeepLearning;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace PFNN_DEV
{
	[RequireComponent(typeof(Actor))]
	public class BioAnimation_F42_v2 : MonoBehaviour
	{

		public bool Inspect = false;

		public bool ShowTrajectory = true;
		public bool ShowVelocities = true;

		public float TargetGain = 0.25f;
		public float TargetDecay = 0.05f;
		public bool TrajectoryControl = true;
		public float TrajectoryCorrection = 1f;


		public int TrajectoryDimIn = 8;
		public int TrajectoryDimOut = 4;
		public int JointDimIn = 6;
		public int JointDimOut = 6;

		public Controller Controller;

		private Actor Actor;
		private PFNN NN;
		private Trajectory Trajectory;

		private Vector3 TargetDirection;
		private Vector3 TargetVelocity;

		//State
		private Vector3[] Positions = new Vector3[0];
		private Vector3[] Forwards = new Vector3[0];
		private Vector3[] Ups = new Vector3[0];
		private Vector3[] Velocities = new Vector3[0];

		//Trajectory for 60 Hz framerate
		private const int Framerate = 60;
		private const int Points = 111;
		private const int PointSamples = 12;
		private const int PastPoints = 60;
		private const int FuturePoints = 50;
		private const int RootPointIndex = 60;
		private const int PointDensity = 10;

		public GameObject JumpTarget;

		void Reset()
		{
			Controller = new Controller();
		}

		void Awake()
		{
			Actor = GetComponent<Actor>();
			NN = GetComponent<PFNN>();
			TargetDirection = new Vector3(transform.forward.x, 0f, transform.forward.z);
			TargetVelocity = Vector3.zero;
			Positions = new Vector3[Actor.Bones.Length];
			Forwards = new Vector3[Actor.Bones.Length];
			Ups = new Vector3[Actor.Bones.Length];
			Velocities = new Vector3[Actor.Bones.Length];
			Trajectory = new Trajectory(Points, Controller.GetNames(), transform.position, TargetDirection);
			if (Controller.Styles.Length > 0)
			{
				for (int i = 0; i < Trajectory.Points.Length; i++)
				{
					Trajectory.Points[i].Styles[0] = 1f;
				}
			}
			for (int i = 0; i < Actor.Bones.Length; i++)
			{
				Positions[i] = Actor.Bones[i].Transform.position;
				Forwards[i] = Actor.Bones[i].Transform.forward;
				Ups[i] = Actor.Bones[i].Transform.up;
				Velocities[i] = Vector3.zero;
			}

			if (NN.Parameters == null)
			{
				Debug.Log("No parameters saved.");
				return;
			}
			NN.LoadParameters();
		}

		void Start()
		{
			Utility.SetFPS(60);
			if (JumpTarget == null)
			{
				JumpTarget = GameObject.FindWithTag("JumpTarget");
			}
		}

		void Update()
		{
			if (NN.Parameters == null)
			{
				return;
			}

			if (TrajectoryControl)
			{
				PredictTrajectory();
			}

			if (NN.Parameters != null)
			{
				Animate();
			}

			transform.position = Trajectory.Points[RootPointIndex].GetPosition();
		}

		private void PredictTrajectory()
		{
			//Calculate Bias
			float bias = PoolBias();

			//Determine Control
			float turn = Controller.QueryTurn();
			Vector3 move = Controller.QueryMove();
			bool control = turn != 0f || move != Vector3.zero;

			//Update Target Direction / Velocity / Correction
			TargetDirection = Vector3.Lerp(TargetDirection, Quaternion.AngleAxis(turn * 60f, Vector3.up) * Trajectory.Points[RootPointIndex].GetDirection(), control ? TargetGain : TargetDecay);
			TargetVelocity = Vector3.Lerp(TargetVelocity, bias * (Quaternion.LookRotation(TargetDirection, Vector3.up) * move).normalized, control ? TargetGain : TargetDecay);
			TrajectoryCorrection = Utility.Interpolate(TrajectoryCorrection, Mathf.Max(move.normalized.magnitude, Mathf.Abs(turn)), control ? TargetGain : TargetDecay);

			//Predict Future Trajectory
			Vector3[] trajectory_positions_blend = new Vector3[Trajectory.Points.Length];
			trajectory_positions_blend[RootPointIndex] = Trajectory.Points[RootPointIndex].GetTransformation().GetPosition();
			for (int i = RootPointIndex + 1; i < Trajectory.Points.Length; i++)
			{
				float bias_pos = 0.75f;
				float bias_dir = 1.25f;
				float bias_vel = 1.50f;
				float weight = (float)(i - RootPointIndex) / (float)FuturePoints; //w between 1/FuturePoints and 1
				float scale_pos = 1.0f - Mathf.Pow(1.0f - weight, bias_pos);
				float scale_dir = 1.0f - Mathf.Pow(1.0f - weight, bias_dir);
				float scale_vel = 1.0f - Mathf.Pow(1.0f - weight, bias_vel);

				float scale = 1f / (Trajectory.Points.Length - (RootPointIndex + 1f));

				trajectory_positions_blend[i] = trajectory_positions_blend[i - 1] +
					Vector3.Lerp(
					Trajectory.Points[i].GetPosition() - Trajectory.Points[i - 1].GetPosition(),
					scale * TargetVelocity,
					scale_pos
					);

				Trajectory.Points[i].SetDirection(Vector3.Lerp(Trajectory.Points[i].GetDirection(), TargetDirection, scale_dir));
				Trajectory.Points[i].SetVelocity(Vector3.Lerp(Trajectory.Points[i].GetVelocity(), TargetVelocity, scale_vel));
			}
			for (int i = RootPointIndex + 1; i < Trajectory.Points.Length; i++)
			{
				Trajectory.Points[i].SetPosition(trajectory_positions_blend[i]);
			}

			float[] style = Controller.GetStyle();
			// if(style[2] == 0f) {
			// 	style[1] = Mathf.Max(style[1], Mathf.Clamp(Trajectory.Points[RootPointIndex].GetVelocity().magnitude, 0f, 1f));
			// }
			// TODO IF ANY OF THE POINTS ARE TOUCHING CUBE THE CHANGE the style array above to be 1 for jump

			// Iterate over all points in the trajectory starting from RootPointIndex
			for (int i = RootPointIndex; i < Trajectory.Points.Length; i++)
			{

				float yDiff = Trajectory.Points[i].GetPosition().y - Trajectory.Points[i - 1].GetPosition().y;
				if (Math.Abs(yDiff) > 0.001f)
				{
					// Debug.Log("Jump!");
					style[0] = 0f;
					style[1] = 0f;
					style[2] = 0.5f;
					break;

				}
			}

			for (int i = RootPointIndex; i < Trajectory.Points.Length; i++)
			{
				float weight = (float)(i - RootPointIndex) / (float)FuturePoints; //w between 0 and 1

				// float yDiff = Trajectory.Points[i].GetPosition().y - Trajectory.Points[i - 1].GetPosition().y;

				// Debug.Log(yDiff);
				// if (Math.Abs(yDiff) > 0.001f)
				// {
				// 	// Debug.Log("Jump!");
				// 	style[0] = 0f;
				// 	style[1] = 0f;
				// 	style[2] = 1f;

				// }

				for (int j = 0; j < Trajectory.Points[i].Styles.Length; j++)
				{
					Trajectory.Points[i].Styles[j] = Utility.Interpolate(Trajectory.Points[i].Styles[j], style[j], Utility.Normalise(weight, 0f, 1f, Controller.Styles[j].Transition, 1f));
				}


				Utility.Normalise(ref Trajectory.Points[i].Styles);
				Trajectory.Points[i].SetSpeed(Utility.Interpolate(Trajectory.Points[i].GetSpeed(), TargetVelocity.magnitude, control ? TargetGain : TargetDecay));
			}
		}

		private void Animate()
		{
			//Calculate Root
			Matrix4x4 currentRoot = Trajectory.Points[RootPointIndex].GetTransformation();
			// currentRoot[1,3] = 0f; //For flat terrain

			int start = 0;
			//Input Trajectory Positions / Directions / Velocities / Styles
			for (int i = 0; i < PointSamples; i++)
			{
				Vector3 pos = GetSample(i).GetPosition().GetRelativePositionTo(currentRoot);
				Vector3 dir = GetSample(i).GetDirection().GetRelativeDirectionTo(currentRoot);
				float slope = GetSample(i).GetSlope();
				NN.SetInput(start + i * TrajectoryDimIn + 0, pos.x);
				NN.SetInput(start + i * TrajectoryDimIn + 1, pos.y);
				NN.SetInput(start + i * TrajectoryDimIn + 2, pos.z);
				NN.SetInput(start + i * TrajectoryDimIn + 3, dir.x);
				NN.SetInput(start + i * TrajectoryDimIn + 4, dir.z);
				NN.SetInput(start + i * TrajectoryDimIn + 5, slope);
				for (int j = 0; j < Controller.Styles.Length; j++)
				{
					NN.SetInput(start + i * TrajectoryDimIn + (TrajectoryDimIn - Controller.Styles.Length) + j, GetSample(i).Styles[j]);
				}
			}
			start += TrajectoryDimIn * PointSamples;

			Matrix4x4 previousRoot = Trajectory.Points[RootPointIndex - 1].GetTransformation();
			// previousRoot[1,3] = 0f; //For flat terrain

			//Input Previous Bone Positions / Velocities
			for (int i = 0; i < Actor.Bones.Length; i++)
			{
				Vector3 pos = Positions[i].GetRelativePositionTo(previousRoot);
				Vector3 vel = Velocities[i].GetRelativeDirectionTo(previousRoot);
				NN.SetInput(start + i * JointDimIn + 0, pos.x);
				NN.SetInput(start + i * JointDimIn + 1, pos.y);
				NN.SetInput(start + i * JointDimIn + 2, pos.z);
				NN.SetInput(start + i * JointDimIn + 3, vel.x);
				NN.SetInput(start + i * JointDimIn + 4, vel.y);
				NN.SetInput(start + i * JointDimIn + 5, vel.z);
			}
			start += JointDimIn * Actor.Bones.Length;

			//Predict
			float rest = Mathf.Pow(1.0f - Trajectory.Points[RootPointIndex].Styles[0], 0.25f);
			((PFNN)NN).SetDamping(1f - (rest * 0.9f + 0.1f));
			NN.Predict();

			//Update Past Trajectory
			for (int i = 0; i < RootPointIndex; i++)
			{
				Trajectory.Points[i].SetPosition(Trajectory.Points[i + 1].GetPosition());
				Trajectory.Points[i].SetDirection(Trajectory.Points[i + 1].GetDirection());
				Trajectory.Points[i].SetVelocity(Trajectory.Points[i + 1].GetVelocity());
				Trajectory.Points[i].SetSpeed(Trajectory.Points[i + 1].GetSpeed());
				Trajectory.Points[i].Postprocess();
				for (int j = 0; j < Trajectory.Points[i].Styles.Length; j++)
				{
					Trajectory.Points[i].Styles[j] = Trajectory.Points[i + 1].Styles[j];
				}
			}

			//Update Root
			Vector3 translationalOffset = Vector3.zero;
			float rotationalOffset = 0f;
			Vector3 rootMotion = new Vector3(NN.GetOutput(TrajectoryDimOut * 6 + JointDimOut * Actor.Bones.Length + 0), NN.GetOutput(TrajectoryDimOut * 6 + JointDimOut * Actor.Bones.Length + 1), NN.GetOutput(TrajectoryDimOut * 6 + JointDimOut * Actor.Bones.Length + 2));
			// rootMotion /= Framerate;
			translationalOffset = rest * new Vector3(rootMotion.x, 0f, rootMotion.z);
			rotationalOffset = rest * rootMotion.y;

			Trajectory.Points[RootPointIndex].SetPosition(translationalOffset.GetRelativePositionFrom(currentRoot));
			Trajectory.Points[RootPointIndex].SetDirection(Quaternion.AngleAxis(rotationalOffset, Vector3.up) * Trajectory.Points[RootPointIndex].GetDirection());
			Trajectory.Points[RootPointIndex].SetVelocity(translationalOffset.GetRelativeDirectionFrom(currentRoot) * Framerate);
			Matrix4x4 nextRoot = Trajectory.Points[RootPointIndex].GetTransformation();
			Trajectory.Points[RootPointIndex].Postprocess();
			// nextRoot[1,3] = 0f; //For flat terrain

			//Update Future Trajectory
			for (int i = RootPointIndex + 1; i < Trajectory.Points.Length; i++)
			{
				Trajectory.Points[i].SetPosition(Trajectory.Points[i].GetPosition() + rest * translationalOffset.GetRelativeDirectionFrom(nextRoot));
				Trajectory.Points[i].SetDirection(Quaternion.AngleAxis(rotationalOffset, Vector3.up) * Trajectory.Points[i].GetDirection());
				Trajectory.Points[i].SetVelocity(Trajectory.Points[i].GetVelocity() + translationalOffset.GetRelativeDirectionFrom(nextRoot) * Framerate);
			}
			start = 0;
			for (int i = RootPointIndex + 1; i < Trajectory.Points.Length; i++)
			{
				//ROOT	1		2		3		4		5
				//.x....x.......x.......x.......x.......x
				int index = i;
				int prevSampleIndex = GetPreviousSample(index).GetIndex() / PointDensity;
				int nextSampleIndex = GetNextSample(index).GetIndex() / PointDensity;
				float factor = (float)(i % PointDensity) / PointDensity;

				Vector3 prevPos = new Vector3(
					NN.GetOutput(start + (prevSampleIndex - 6) * TrajectoryDimOut + 0),
					0f,
					NN.GetOutput(start + (prevSampleIndex - 6) * TrajectoryDimOut + 1)
				).GetRelativePositionFrom(nextRoot);
				Vector3 prevDir = new Vector3(
					NN.GetOutput(start + (prevSampleIndex - 6) * TrajectoryDimOut + 2),
					0f,
					NN.GetOutput(start + (prevSampleIndex - 6) * TrajectoryDimOut + 3)
				).normalized.GetRelativeDirectionFrom(nextRoot);

				Vector3 nextPos = new Vector3(
					NN.GetOutput(start + (nextSampleIndex - 6) * TrajectoryDimOut + 0),
					0f,
					NN.GetOutput(start + (nextSampleIndex - 6) * TrajectoryDimOut + 1)
				).GetRelativePositionFrom(nextRoot);
				Vector3 nextDir = new Vector3(
					NN.GetOutput(start + (nextSampleIndex - 6) * TrajectoryDimOut + 2),
					0f,
					NN.GetOutput(start + (nextSampleIndex - 6) * TrajectoryDimOut + 3)
				).normalized.GetRelativeDirectionFrom(nextRoot);

				Vector3 pos = (1f - factor) * prevPos + factor * nextPos;
				Vector3 dir = ((1f - factor) * prevDir + factor * nextDir).normalized;

				pos = Vector3.Lerp(Trajectory.Points[i].GetPosition(), pos, 0.5f);

				Trajectory.Points[i].SetPosition(
					Utility.Interpolate(
						Trajectory.Points[i].GetPosition(),
						pos,
						TrajectoryCorrection
						)
					);
				Trajectory.Points[i].SetDirection(
					Utility.Interpolate(
						Trajectory.Points[i].GetDirection(),
						dir,
						TrajectoryCorrection
						)
					);

				Trajectory.Points[i].Postprocess();

			}
			start += TrajectoryDimOut * 6;

			//Compute Posture
			for (int i = 0; i < Actor.Bones.Length; i++)
			{
				Vector3 position = new Vector3(NN.GetOutput(start + i * JointDimOut + 0), NN.GetOutput(start + i * JointDimOut + 1), NN.GetOutput(start + i * JointDimOut + 2)).GetRelativePositionFrom(currentRoot);
				Vector3 velocity = new Vector3(NN.GetOutput(start + i * JointDimOut + 3), NN.GetOutput(start + i * JointDimOut + 4), NN.GetOutput(start + i * JointDimOut + 5)).GetRelativeDirectionFrom(currentRoot);

				Positions[i] = Vector3.Lerp(Positions[i] + velocity / Framerate, position, 0.5f);
				Velocities[i] = velocity;
			}
			start += JointDimOut * Actor.Bones.Length;

			//Assign Posture
			transform.position = nextRoot.GetPosition();
			transform.rotation = nextRoot.GetRotation();
			for (int i = 0; i < Actor.Bones.Length; i++)
			{
				Actor.Bones[i].Transform.position = Positions[i];
				Actor.Bones[i].Transform.rotation = Quaternion.LookRotation(Forwards[i], Ups[i]);
			}
		}

		private float PoolBias()
		{
			float[] styles = Trajectory.Points[RootPointIndex].Styles;
			float bias = 0f;
			for (int i = 0; i < styles.Length; i++)
			{
				float _bias = Controller.Styles[i].Bias;
				float max = 0f;
				for (int j = 0; j < Controller.Styles[i].Multipliers.Length; j++)
				{
					if (Input.GetKey(Controller.Styles[i].Multipliers[j].Key))
					{
						max = Mathf.Max(max, Controller.Styles[i].Bias * Controller.Styles[i].Multipliers[j].Value);
					}
				}
				for (int j = 0; j < Controller.Styles[i].Multipliers.Length; j++)
				{
					if (Input.GetKey(Controller.Styles[i].Multipliers[j].Key))
					{
						_bias = Mathf.Min(max, _bias * Controller.Styles[i].Multipliers[j].Value);
					}
				}
				bias += styles[i] * _bias;
			}
			return bias;
		}

		private Trajectory.Point GetSample(int index)
		{
			return Trajectory.Points[Mathf.Clamp(index * 10, 0, Trajectory.Points.Length - 1)];
		}

		private Trajectory.Point GetPreviousSample(int index)
		{
			return GetSample(index / 10);
		}

		private Trajectory.Point GetNextSample(int index)
		{
			if (index % 10 == 0)
			{
				return GetSample(index / 10);
			}
			else
			{
				return GetSample(index / 10 + 1);
			}
		}

		void OnRenderObject()
		{
			if (Application.isPlaying)
			{
				if (NN.Parameters == null)
				{
					return;
				}

				if (ShowTrajectory)
				{
					UltiDraw.Begin();
					// TODO COMMENT BACK IN IF WANT DIRECTION
					// UltiDraw.DrawLine(Trajectory.Points[RootPointIndex].GetPosition(), Trajectory.Points[RootPointIndex].GetPosition() + TargetDirection, 0.05f, 0f, UltiDraw.Red.Transparent(0.75f));
					// TODO THIS USED TO BE GREEN
					UltiDraw.DrawLine(Trajectory.Points[RootPointIndex].GetPosition(), Trajectory.Points[RootPointIndex].GetPosition() + TargetVelocity, 0.05f, 0f, UltiDraw.Red.Transparent(0.75f));
					UltiDraw.End();
					Trajectory.Draw(10);
				}

				if (ShowVelocities)
				{
					UltiDraw.Begin();
					for (int i = 0; i < Actor.Bones.Length; i++)
					{
						UltiDraw.DrawArrow(
							Actor.Bones[i].Transform.position,
							Actor.Bones[i].Transform.position + Velocities[i],
							0.75f,
							0.0075f,
							0.05f,
							UltiDraw.Purple.Transparent(0.5f)
						);
					}
					UltiDraw.End();
				}

				UltiDraw.Begin();
				// UltiDraw.DrawGUIHorizontalBar(new Vector2(0.5f, 0.9f), new Vector2(0.25f, 0.05f), UltiDraw.White, 0.0025f, UltiDraw.Mustard, NN.GetPhase(), UltiDraw.DarkGrey);
				UltiDraw.End();
			}
		}

		void OnDrawGizmos()
		{
			if (!Application.isPlaying)
			{
				OnRenderObject();
			}
		}

#if UNITY_EDITOR
		[CustomEditor(typeof(BioAnimation_F42_v2))]
		public class BioAnimation_F42_v2_Editor : Editor
		{

			public BioAnimation_F42_v2 Target;

			void Awake()
			{
				Target = (BioAnimation_F42_v2)target;
			}

			public override void OnInspectorGUI()
			{
				Undo.RecordObject(Target, Target.name);

				Inspector();
				Target.Controller.Inspector();

				if (GUI.changed)
				{
					EditorUtility.SetDirty(Target);
				}
			}

			private void Inspector()
			{
				Utility.SetGUIColor(UltiDraw.Grey);
				using (new EditorGUILayout.VerticalScope("Box"))
				{
					Utility.ResetGUIColor();

					if (Utility.GUIButton("Animation", UltiDraw.DarkGrey, UltiDraw.White))
					{
						Target.Inspect = !Target.Inspect;
					}

					if (Target.Inspect)
					{
						using (new EditorGUILayout.VerticalScope("Box"))
						{
							Target.TrajectoryDimIn = EditorGUILayout.IntField("Trajectory Dim X", Target.TrajectoryDimIn);
							Target.TrajectoryDimOut = EditorGUILayout.IntField("Trajectory Dim Y", Target.TrajectoryDimOut);
							Target.JointDimIn = EditorGUILayout.IntField("Joint Dim X", Target.JointDimIn);
							Target.JointDimOut = EditorGUILayout.IntField("Joint Dim Y", Target.JointDimOut);
							Target.ShowTrajectory = EditorGUILayout.Toggle("Show Trajectory", Target.ShowTrajectory);
							Target.ShowVelocities = EditorGUILayout.Toggle("Show Velocities", Target.ShowVelocities);
							Target.TargetGain = EditorGUILayout.Slider("Target Gain", Target.TargetGain, 0f, 1f);
							Target.TargetDecay = EditorGUILayout.Slider("Target Decay", Target.TargetDecay, 0f, 1f);
							Target.TrajectoryControl = EditorGUILayout.Toggle("Trajectory Control", Target.TrajectoryControl);
							Target.TrajectoryCorrection = EditorGUILayout.Slider("Trajectory Correction", Target.TrajectoryCorrection, 0f, 1f);
						}
					}
				}
			}
		}
#endif
	}
}