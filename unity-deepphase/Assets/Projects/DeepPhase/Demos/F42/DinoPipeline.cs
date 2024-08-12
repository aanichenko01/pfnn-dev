#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.Collections;
using System.Collections.Generic;
using AI4Animation;

namespace DeepPhase
{
    public class DinoPipeline : AssetPipelineSetup
    {

        public enum MODE { ProcessAssets, ExportController, ExportControllerLMP, ExportControllerMANN };
        public MODE Mode = MODE.ProcessAssets;

        public int Channels = 5;
        public bool WriteMirror = true;

        private DateTime Timestamp;
        private float Progress = 0f;
        private float SamplesPerSecond = 0f;
        private int Samples = 0;
        private int Sequence = 0;

        private AssetPipeline.Data.File S;
        private AssetPipeline.Data X, Y;

        public override void Inspector()
        {
            Mode = (MODE)EditorGUILayout.EnumPopup("Mode", Mode);
            void ExportMode()
            {
                EditorGUI.BeginDisabledGroup(true);
                EditorGUILayout.FloatField("Export Framerate", Pipeline.GetEditor().TargetFramerate);
                EditorGUILayout.TextField("Export Path", AssetPipeline.Data.GetExportPath());
                EditorGUI.EndDisabledGroup();
                if (Mode == MODE.ExportController)
                {
                    Channels = EditorGUILayout.IntField("Channels", Channels);
                }
                WriteMirror = EditorGUILayout.Toggle("Write Mirror", WriteMirror);
                if (Pipeline.IsProcessing() || Pipeline.IsAborting())
                {
                    EditorGUI.BeginDisabledGroup(true);
                    EditorGUILayout.FloatField("Samples Per Second", SamplesPerSecond);
                    EditorGUI.EndDisabledGroup();
                    EditorGUI.DrawRect(
                        new Rect(
                            EditorGUILayout.GetControlRect().x,
                            EditorGUILayout.GetControlRect().y,
                            Progress * EditorGUILayout.GetControlRect().width, 25f
                        ),
                        UltiDraw.Green.Opacity(0.75f)
                    );
                }
            }
            if (Mode == MODE.ProcessAssets)
            {
                //Nothing to do
            }
            if (Mode == MODE.ExportController)
            {
                ExportMode();
            }
            if (Mode == MODE.ExportControllerLMP)
            {
                ExportMode();
            }
            if (Mode == MODE.ExportControllerMANN)
            {
                ExportMode();
            }
        }

        public override void Inspector(AssetPipeline.Item item)
        {

        }

        public override bool CanProcess()
        {
            return true;
        }

        public override void Begin()
        {
            if (Mode == MODE.ProcessAssets)
            {
                //Nothing to do
            }
            if (Mode == MODE.ExportController)
            {
                Samples = 0;
                Sequence = 0;
                S = AssetPipeline.Data.CreateFile("Sequences", AssetPipeline.Data.TYPE.Text);
                X = new AssetPipeline.Data("Input");
                Y = new AssetPipeline.Data("Output");
            }
            if (Mode == MODE.ExportControllerLMP)
            {
                Samples = 0;
                Sequence = 0;
                S = AssetPipeline.Data.CreateFile("Sequences", AssetPipeline.Data.TYPE.Text);
                X = new AssetPipeline.Data("Input");
                Y = new AssetPipeline.Data("Output");
            }
            if (Mode == MODE.ExportControllerMANN)
            {
                Samples = 0;
                Sequence = 0;
                S = AssetPipeline.Data.CreateFile("Sequences", AssetPipeline.Data.TYPE.Text);
                X = new AssetPipeline.Data("Input");
                Y = new AssetPipeline.Data("Output");
            }
        }

        private void WriteSequenceInfo(int sequence, float timestamp, bool mirrored, MotionAsset asset)
        {
            //Sequence - Timestamp - Mirroring - Name - GUID
            S.WriteLine(
                sequence.ToString() + AssetPipeline.Data.Separator +
                timestamp + AssetPipeline.Data.Separator +
                (mirrored ? "Mirrored" : "Standard") + AssetPipeline.Data.Separator +
                asset.name + AssetPipeline.Data.Separator +
                Utility.GetAssetGUID(asset));
        }

        public override IEnumerator Iterate(MotionAsset asset)
        {
            Pipeline.GetEditor().LoadSession(Utility.GetAssetGUID(asset));
            if (Mode == MODE.ProcessAssets)
            {
                ProcessAssets(asset);
            }
            if (Mode == MODE.ExportController)
            {
                if (asset.Export)
                {
                    for (int i = 1; i <= 2; i++)
                    {
                        if (i == 1)
                        {
                            Pipeline.GetEditor().SetMirror(false);
                        }
                        else if (i == 2 && WriteMirror)
                        {
                            Pipeline.GetEditor().SetMirror(true);
                        }
                        else
                        {
                            break;
                        }
                        foreach (Interval seq in asset.Sequences)
                        {
                            Sequence += 1;
                            float start = asset.GetFrame(asset.GetFrame(seq.Start).Timestamp).Timestamp;
                            float end = asset.GetFrame(asset.GetFrame(seq.End).Timestamp - 1f / Pipeline.GetEditor().TargetFramerate).Timestamp;
                            int index = 0;
                            while (Pipeline.IsProcessing() && (start + index / Pipeline.GetEditor().TargetFramerate < end || Mathf.Approximately(start + index / Pipeline.GetEditor().TargetFramerate, end)))
                            {
                                float tCurrent = start + index / Pipeline.GetEditor().TargetFramerate;
                                float tNext = start + (index + 1) / Pipeline.GetEditor().TargetFramerate;
                                index += 1;

                                ControllerSetup.Export(this, X, Y, tCurrent, tNext);

                                X.Store();
                                Y.Store();
                                WriteSequenceInfo(Sequence, tCurrent, Pipeline.GetEditor().Mirror, asset);

                                Samples += 1;
                                if (Utility.GetElapsedTime(Timestamp) >= 0.1f)
                                {
                                    Progress = end == 0f ? 1f : (tCurrent / end);
                                    SamplesPerSecond = Samples / (float)Utility.GetElapsedTime(Timestamp);
                                    Samples = 0;
                                    Timestamp = Utility.GetTimestamp();
                                    yield return new WaitForSeconds(0f);
                                }
                            }
                        }
                    }
                }
            }
            if (Mode == MODE.ExportControllerLMP)
            {
                if (asset.Export)
                {
                    for (int i = 1; i <= 2; i++)
                    {
                        if (i == 1)
                        {
                            Pipeline.GetEditor().SetMirror(false);
                        }
                        else if (i == 2 && WriteMirror)
                        {
                            Pipeline.GetEditor().SetMirror(true);
                        }
                        else
                        {
                            break;
                        }
                        foreach (Interval seq in asset.Sequences)
                        {
                            Sequence += 1;
                            float start = asset.GetFrame(asset.GetFrame(seq.Start).Timestamp).Timestamp;
                            float end = asset.GetFrame(asset.GetFrame(seq.End).Timestamp - 1f / Pipeline.GetEditor().TargetFramerate).Timestamp;
                            int index = 0;
                            while (Pipeline.IsProcessing() && (start + index / Pipeline.GetEditor().TargetFramerate < end || Mathf.Approximately(start + index / Pipeline.GetEditor().TargetFramerate, end)))
                            {
                                float tCurrent = start + index / Pipeline.GetEditor().TargetFramerate;
                                float tNext = start + (index + 1) / Pipeline.GetEditor().TargetFramerate;
                                index += 1;

                                ControllerLMPSetup.Export(this, X, Y, tCurrent, tNext);

                                X.Store();
                                Y.Store();
                                WriteSequenceInfo(Sequence, tCurrent, Pipeline.GetEditor().Mirror, asset);

                                Samples += 1;
                                if (Utility.GetElapsedTime(Timestamp) >= 0.1f)
                                {
                                    Progress = end == 0f ? 1f : (tCurrent / end);
                                    SamplesPerSecond = Samples / (float)Utility.GetElapsedTime(Timestamp);
                                    Samples = 0;
                                    Timestamp = Utility.GetTimestamp();
                                    yield return new WaitForSeconds(0f);
                                }
                            }
                        }
                    }
                }
            }
            if (Mode == MODE.ExportControllerMANN)
            {
                if (asset.Export)
                {
                    for (int i = 1; i <= 2; i++)
                    {
                        if (i == 1)
                        {
                            Pipeline.GetEditor().SetMirror(false);
                        }
                        else if (i == 2 && WriteMirror)
                        {
                            Pipeline.GetEditor().SetMirror(true);
                        }
                        else
                        {
                            break;
                        }
                        foreach (Interval seq in asset.Sequences)
                        {
                            Sequence += 1;
                            float start = asset.GetFrame(asset.GetFrame(seq.Start).Timestamp).Timestamp;
                            float end = asset.GetFrame(asset.GetFrame(seq.End).Timestamp - 1f / Pipeline.GetEditor().TargetFramerate).Timestamp;
                            int index = 0;
                            while (Pipeline.IsProcessing() && (start + index / Pipeline.GetEditor().TargetFramerate < end || Mathf.Approximately(start + index / Pipeline.GetEditor().TargetFramerate, end)))
                            {
                                float tCurrent = start + index / Pipeline.GetEditor().TargetFramerate;
                                float tNext = start + (index + 1) / Pipeline.GetEditor().TargetFramerate;
                                index += 1;

                                ControllerMANNSetup.Export(this, X, Y, tCurrent, tNext);

                                X.Store();
                                Y.Store();
                                WriteSequenceInfo(Sequence, tCurrent, Pipeline.GetEditor().Mirror, asset);

                                Samples += 1;
                                if (Utility.GetElapsedTime(Timestamp) >= 0.1f)
                                {
                                    Progress = end == 0f ? 1f : (tCurrent / end);
                                    SamplesPerSecond = Samples / (float)Utility.GetElapsedTime(Timestamp);
                                    Samples = 0;
                                    Timestamp = Utility.GetTimestamp();
                                    yield return new WaitForSeconds(0f);
                                }
                            }
                        }
                    }
                }
            }
        }

        public override void Callback()
        {
            if (Mode == MODE.ProcessAssets)
            {
                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();
                Resources.UnloadUnusedAssets();
            }
            if (Mode == MODE.ExportController)
            {
                Resources.UnloadUnusedAssets();
            }
            if (Mode == MODE.ExportControllerLMP)
            {
                Resources.UnloadUnusedAssets();
            }
            if (Mode == MODE.ExportControllerMANN)
            {
                Resources.UnloadUnusedAssets();
            }
        }

        public override void Finish()
        {
            if (Mode == MODE.ProcessAssets)
            {
                for (int i = 0; i < Pipeline.GetEditor().Assets.Count; i++)
                {
                    //MotionAsset.Retrieve(Pipeline.GetEditor().Assets[i]).ResetSequences();
                    MotionAsset.Retrieve(Pipeline.GetEditor().Assets[i]).Export = false;
                }

                // TOD

                // Idle to Walk
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[0]).Export = true;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[0]).SetSequence(0, 1, 125);

                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[1]).Export = true;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[1]).SetSequence(0, 1, 132);

                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[2]).Export = true;
                MotionAsset.Retrieve(Pipeline.GetEditor().Assets[2]).SetSequence(0, 1, 124);

                for (int i = 0; i < Pipeline.GetEditor().Assets.Count; i++)
                {
                    MotionAsset.Retrieve(Pipeline.GetEditor().Assets[i]).MarkDirty(true, false);
                }
                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();
                Resources.UnloadUnusedAssets();
            }
            if (Mode == MODE.ExportController)
            {
                S.Close();
                X.Finish();
                Y.Finish();
            }
            if (Mode == MODE.ExportControllerLMP)
            {
                S.Close();
                X.Finish();
                Y.Finish();
            }
            if (Mode == MODE.ExportControllerMANN)
            {
                S.Close();
                X.Finish();
                Y.Finish();
            }
        }

        private void ProcessAssets(MotionAsset asset)
        {
            asset.RemoveAllModules();

            asset.MirrorAxis = Axis.ZPositive;
            asset.Model = "AnzuBaby_RIG_DME_0005a_BINARY";
            asset.Scale = 1f;
            asset.Source.FindBone("AnzB:Head").Alignment = new Vector3(0f, 0f, 0f);
            asset.Source.FindBone("AnzB:LeftShoulder").Alignment = new Vector3(0f, 0f, 0f);
            asset.Source.FindBone("AnzB:RightShoulder").Alignment = new Vector3(0f, 0f, 0f);
            asset.FootContactReferenceIdx = asset.Source.FindBone("AnzB:LeftFootIndex4").Index;

            if (asset.name.Contains("Jump"))
            {
                asset.AddJumpTarget = true;
            }

            {
                RootModule module = asset.HasModule<RootModule>() ? asset.GetModule<RootModule>() : asset.AddModule<RootModule>();
                module.Topology = RootModule.TOPOLOGY.Biped;
                module.SmoothRotations = true;
            }

            {
                ContactModule module = asset.HasModule<ContactModule>() ? asset.GetModule<ContactModule>() : asset.AddModule<ContactModule>();
                module.Clear();
                module.AddSensor("AnzB:Hips", Vector3.zero, Vector3.zero, 0.2f * Vector3.one, 1f, LayerMask.GetMask("Ground"), ContactModule.ContactType.Translational, ContactModule.ColliderType.Sphere);
                module.AddSensor("AnzB:Neck", Vector3.zero, Vector3.zero, 0.25f * Vector3.one, 1f, LayerMask.GetMask("Ground"), ContactModule.ContactType.Translational, ContactModule.ColliderType.Sphere);
                module.AddSensor("AnzB:LeftHand", Vector3.zero, Vector3.zero, 1f / 30f * Vector3.one, 1f, LayerMask.GetMask("Ground"), ContactModule.ContactType.Translational, ContactModule.ColliderType.Sphere);
                module.AddSensor("AnzB:RightHand", Vector3.zero, Vector3.zero, 1f / 30f * Vector3.one, 1f, LayerMask.GetMask("Ground"), ContactModule.ContactType.Translational, ContactModule.ColliderType.Sphere);
                module.AddSensor("AnzB:LeftFootIndex4", Vector3.zero, Vector3.zero, 1f / 30f * Vector3.one, 1f, LayerMask.GetMask("Ground"), ContactModule.ContactType.Translational, ContactModule.ColliderType.Sphere);
                module.AddSensor("AnzB:RightFootIndex4", Vector3.zero, Vector3.zero, 1f / 30f * Vector3.one, 1f, LayerMask.GetMask("Ground"), ContactModule.ContactType.Translational, ContactModule.ColliderType.Sphere);
                module.CaptureContacts(Pipeline.GetEditor());
            }

            {
                StyleModule module = asset.HasModule<StyleModule>() ? asset.GetModule<StyleModule>() : asset.AddModule<StyleModule>();
                module.Clear();

                RootModule root = asset.GetModule<RootModule>();
                StyleModule.Function idle = module.AddFunction("Idle");
                StyleModule.Function move = module.AddFunction("Move");
                StyleModule.Function speed = module.AddFunction("Speed");
                StyleModule.Function jump = module.AddFunction("Jump");
                float threshold = 0.1f;
                float jumpThreshold = 0.2f;
                float[] weights = new float[asset.Frames.Length];
                float[] rootMotion = new float[asset.Frames.Length];
                float[] bodyMotion = new float[asset.Frames.Length];
                float[] rootMotionY = new float[asset.Frames.Length];
                for (int f = 0; f < asset.Frames.Length; f++)
                {
                    rootMotion[f] = root.GetRootVelocity(asset.Frames[f].Timestamp, false).magnitude;
                    bodyMotion[f] = asset.Frames[f].GetBoneVelocities(Pipeline.GetEditor().GetSession().GetBoneMapping(), false).Magnitudes().Mean();
                    rootMotionY[f] = asset.Frames[f].GetBoneVelocities(Pipeline.GetEditor().GetSession().GetBoneMapping(), false).ToArrayY().Mean();

                }
                {
                    float[] copy = rootMotion.Copy();
                    for (int i = 0; i < copy.Length; i++)
                    {
                        rootMotion[i] = copy.GatherByWindow(i, Mathf.RoundToInt(0.5f * root.Window * asset.Framerate)).Gaussian();
                    }
                }
                {
                    float[] copy = bodyMotion.Copy();
                    for (int i = 0; i < copy.Length; i++)
                    {
                        bodyMotion[i] = copy.GatherByWindow(i, Mathf.RoundToInt(0.5f * root.Window * asset.Framerate)).Gaussian();
                    }
                }
                for (int f = 0; f < asset.Frames.Length; f++)
                {
                    float motion = Mathf.Min(rootMotion[f], bodyMotion[f]);
                    float movement = root.GetRootLength(asset.Frames[f].Timestamp, false);
                    float jumpMotion = Mathf.Abs(rootMotionY[f]);
                    Debug.Log($"Frame={f},jumpMotion= {jumpMotion}");
                    idle.StandardValues[f] = motion < threshold ? 1f : 0f;
                    idle.MirroredValues[f] = motion < threshold ? 1f : 0f;
                    move.StandardValues[f] = 1f - idle.StandardValues[f];
                    move.MirroredValues[f] = 1f - idle.StandardValues[f];
                    speed.StandardValues[f] = movement;
                    speed.MirroredValues[f] = movement;
                    jump.StandardValues[f] = jumpMotion > jumpThreshold ? 1f : 0f;
                    jump.MirroredValues[f] = jumpMotion > jumpThreshold ? 1f : 0f;
                    weights[f] = Mathf.Sqrt(Mathf.Clamp(motion, 0f, threshold).Normalize(0f, threshold, 0f, 1f));
                }
                {
                    float[] copy = idle.StandardValues.Copy();
                    for (int i = 0; i < copy.Length; i++)
                    {
                        idle.StandardValues[i] = copy.GatherByWindow(i, Mathf.RoundToInt(weights[i] * 0.5f * root.Window * asset.Framerate)).Gaussian().SmoothStep(2f, 0.5f);
                        idle.MirroredValues[i] = copy.GatherByWindow(i, Mathf.RoundToInt(weights[i] * 0.5f * root.Window * asset.Framerate)).Gaussian().SmoothStep(2f, 0.5f);
                    }
                }
                {
                    float[] copy = move.StandardValues.Copy();
                    for (int i = 0; i < copy.Length; i++)
                    {
                        move.StandardValues[i] = copy.GatherByWindow(i, Mathf.RoundToInt(weights[i] * 0.5f * root.Window * asset.Framerate)).Gaussian().SmoothStep(2f, 0.5f);
                        move.MirroredValues[i] = copy.GatherByWindow(i, Mathf.RoundToInt(weights[i] * 0.5f * root.Window * asset.Framerate)).Gaussian().SmoothStep(2f, 0.5f);
                    }
                }
                {
                    float[] copy = speed.StandardValues.Copy();
                    float[] grads = copy.Gradients(asset.GetDeltaTime());
                    for (int i = 0; i < speed.StandardValues.Length; i++)
                    {
                        int padding = Mathf.RoundToInt(weights[i] * 0.5f * root.Window * asset.Framerate);
                        float power = Mathf.Abs(grads.GatherByWindow(i, padding).Gaussian());
                        speed.StandardValues[i] = copy.GatherByWindow(i, padding).Gaussian(power);
                        speed.StandardValues[i] = Mathf.Lerp(speed.StandardValues[i], 0f, idle.StandardValues[i]);

                        speed.MirroredValues[i] = copy.GatherByWindow(i, padding).Gaussian(power);
                        speed.MirroredValues[i] = Mathf.Lerp(speed.MirroredValues[i], 0f, idle.MirroredValues[i]);
                    }
                }
                {
                    float[] copy = jump.StandardValues.Copy();
                    for (int i = 0; i < copy.Length; i++)
                    {
                        jump.StandardValues[i] = copy.GatherByWindow(i, Mathf.RoundToInt(weights[i] * 0.5f * root.Window * asset.Framerate)).Gaussian().SmoothStep(2f, 0.5f);
                        jump.MirroredValues[i] = copy.GatherByWindow(i, Mathf.RoundToInt(weights[i] * 0.5f * root.Window * asset.Framerate)).Gaussian().SmoothStep(2f, 0.5f);
                    }
                }

            }

            // {
            //     StyleModule module = asset.HasModule<StyleModule>() ? asset.GetModule<StyleModule>() : asset.AddModule<StyleModule>();
            //     module.Clear();

            //     RootModule root = asset.GetModule<RootModule>();
            //     StyleModule.StyleFunction rooted = module.AddStyle("Rooted");
            //     StyleModule.StyleFunction speed = module.AddStyle("Speed");
            //     float threshold = 0.1f;
            //     float[] rootMotion = new float[asset.Frames.Length];
            //     float[] bodyMotion = new float[asset.Frames.Length];
            //     for(int f=0; f<asset.Frames.Length; f++) {
            //         rootMotion[f] = root.GetRootVelocity(asset.Frames[f].Timestamp, false).magnitude;
            //         bodyMotion[f] = asset.Frames[f].GetBoneVelocities(Pipeline.GetEditor().GetSession().GetBoneMapping(), false).Magnitudes().Mean();
            //     }
            //     float GetMotionValue(int index) {
            //         return Mathf.Max(rootMotion[index], bodyMotion[index]);
            //     }
            //     float GetMotionWeight(int index) {
            //         return Mathf.Clamp(GetMotionValue(index), 0f, threshold).Normalize(0f, threshold, 0f, 1f).SmoothStep(2f, 0.5f);
            //     }
            //     for(int f=0; f<asset.Frames.Length; f++) {
            //         rooted.Values[f] = 1f-GetMotionWeight(f);
            //         speed.Values[f] = root.GetTranslationalSpeed(asset.Frames[f].Timestamp, false);
            //     }
            //     rooted.Values.SmoothGaussian(Mathf.RoundToInt(0.5f*root.Window*asset.Framerate));
            //     {
            //         float[] copy = speed.Values.Copy();
            //         float[] grads = copy.Gradients(asset.GetDeltaTime());
            //         for(int i=0; i<copy.Length; i++) {
            //             int padding = Mathf.RoundToInt(0.5f*root.Window*asset.Framerate);
            //             float power = Mathf.Abs(grads.GatherByWindow(i, padding).Gaussian());
            //             speed.Values[i] = Mathf.Lerp(copy.GatherByWindow(i, padding).Gaussian(power), speed.Values[i], 1f-GetMotionWeight(i));
            //         }
            //     }
            // }

            // if(!asset.HasModule<TailModule>()) {
            //     asset.AddModule<TailModule>();
            // }

            asset.MarkDirty(true, false);
        }

        private class ControllerSetup
        {
            public static void Export(DinoPipeline setup, AssetPipeline.Data X, AssetPipeline.Data Y, float tCurrent, float tNext)
            {
                Container current = new Container(setup, tCurrent);
                Container next = new Container(setup, tNext);

                string[] styles = new string[] { "Idle", "Move", "Speed" };
                // string[] styles = new string[]{"Speed"};
                // string[] contacts = new string[]{"LeftHandSite", "RightHandSite", "LeftFootSite", "RightFootSite"};
                string[] contacts = new string[] { "AnzB:LeftFootIndex4", "AnzB:RightFootIndex4" };

                //Input
                //Control
                for (int k = 0; k < current.TimeSeries.Samples.Length; k++)
                {
                    X.FeedXZ(next.RootSeries.GetPosition(k).PositionTo(current.Root), "TrajectoryPosition" + (k + 1));
                    X.FeedXZ(next.RootSeries.GetDirection(k).DirectionTo(current.Root), "TrajectoryDirection" + (k + 1));
                    X.FeedXZ(next.RootSeries.GetVelocity(k).DirectionTo(current.Root), "TrajectoryVelocity" + (k + 1));
                    X.Feed(next.StyleSeries.GetValues(k, styles), "Actions" + (k + 1) + "-");
                }

                //Auto-Regressive Posture
                for (int k = 0; k < current.ActorPosture.Length; k++)
                {
                    X.Feed(current.ActorPosture[k].GetPosition().PositionTo(current.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Position");
                    X.Feed(current.ActorPosture[k].GetForward().DirectionTo(current.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Forward");
                    X.Feed(current.ActorPosture[k].GetUp().DirectionTo(current.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Up");
                    X.Feed(current.ActorVelocities[k].DirectionTo(current.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Velocity");
                }

                //Gating Variables
                X.Feed(current.PhaseSeries.GetAlignment(), "PhaseSpace-");

                //Output
                //Root Update
                Matrix4x4 delta = next.Root.TransformationTo(current.Root);
                Y.Feed(new Vector3(delta.GetPosition().x, Vector3.SignedAngle(Vector3.forward, delta.GetForward(), Vector3.up), delta.GetPosition().z), "RootUpdate");
                Y.FeedXZ(next.RootSeries.Velocities[next.TimeSeries.Pivot].DirectionTo(next.Root), "RootVelocity");
                Y.Feed(next.StyleSeries.GetValues(next.TimeSeries.Pivot, styles), "RootActions");

                //Control
                for (int k = next.TimeSeries.Pivot + 1; k < next.TimeSeries.Samples.Length; k++)
                {
                    Y.FeedXZ(next.RootSeries.GetPosition(k).PositionTo(next.Root), "TrajectoryPosition" + (k + 1));
                    Y.FeedXZ(next.RootSeries.GetDirection(k).DirectionTo(next.Root), "TrajectoryDirection" + (k + 1));
                    Y.FeedXZ(next.RootSeries.GetVelocity(k).DirectionTo(next.Root), "TrajectoryVelocity" + (k + 1));
                    Y.Feed(next.StyleSeries.GetValues(k, styles), "Actions" + (k + 1) + "-");
                }

                //Auto-Regressive Posture
                for (int k = 0; k < next.ActorPosture.Length; k++)
                {
                    Y.Feed(next.ActorPosture[k].GetPosition().PositionTo(next.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Position");
                    Y.Feed(next.ActorPosture[k].GetForward().DirectionTo(next.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Forward");
                    Y.Feed(next.ActorPosture[k].GetUp().DirectionTo(next.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Up");
                    Y.Feed(next.ActorVelocities[k].DirectionTo(next.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Velocity");
                }

                //Contacts
                Y.Feed(next.ContactSeries.GetContacts(next.TimeSeries.Pivot, contacts), "Contacts-");

                //Phase Update
                Y.Feed(next.PhaseSeries.GetUpdate(), "PhaseUpdate-");
            }

            private class Container
            {
                public MotionAsset Asset;
                public Frame Frame;
                public Actor Actor;

                public TimeSeries TimeSeries;
                public RootModule.Series RootSeries;
                public StyleModule.Series StyleSeries;
                public ContactModule.Series ContactSeries;
                public DeepPhaseModule.Series PhaseSeries;

                //Actor Features
                public Matrix4x4 Root;
                public Matrix4x4[] ActorPosture;
                public Vector3[] ActorVelocities;

                public Container(DinoPipeline setup, float timestamp)
                {
                    MotionEditor editor = setup.Pipeline.GetEditor();
                    editor.LoadFrame(timestamp);
                    Asset = editor.GetSession().Asset;
                    Frame = editor.GetCurrentFrame();

                    TimeSeries = editor.GetTimeSeries();
                    RootSeries = Asset.GetModule<RootModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror) as RootModule.Series;
                    StyleSeries = Asset.GetModule<StyleModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror) as StyleModule.Series;
                    ContactSeries = Asset.GetModule<ContactModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror) as ContactModule.Series;
                    PhaseSeries = Asset.GetModule<DeepPhaseModule>(setup.Channels + "Channels").ExtractSeries(TimeSeries, timestamp, editor.Mirror) as DeepPhaseModule.Series;

                    Root = editor.GetSession().GetActor().transform.GetWorldMatrix();
                    ActorPosture = editor.GetSession().GetActor().GetBoneTransformations();
                    ActorVelocities = editor.GetSession().GetActor().GetBoneVelocities();
                }
            }
        }


        private class ControllerLMPSetup
        {
            public static void Export(DinoPipeline setup, AssetPipeline.Data X, AssetPipeline.Data Y, float tCurrent, float tNext)
            {
                Container current = new Container(setup, tCurrent);
                Container next = new Container(setup, tNext);

                string[] styles = new string[] { "Idle", "Move", "Speed" };
                // string[] styles = new string[]{"Speed"};
                // string[] contacts = new string[]{"LeftHandSite", "RightHandSite", "LeftFootSite", "RightFootSite"};
                string[] contacts = new string[] { "AnzB:LeftFootIndex4", "AnzB:RightFootIndex4" };

                //Input
                //Control
                for (int k = 0; k < current.TimeSeries.Samples.Length; k++)
                {
                    X.FeedXZ(next.RootSeries.GetPosition(k).PositionTo(current.Root), "TrajectoryPosition" + (k + 1));
                    X.FeedXZ(next.RootSeries.GetDirection(k).DirectionTo(current.Root), "TrajectoryDirection" + (k + 1));
                    X.FeedXZ(next.RootSeries.GetVelocity(k).DirectionTo(current.Root), "TrajectoryVelocity" + (k + 1));
                    X.Feed(next.StyleSeries.GetValues(k, styles), "Actions" + (k + 1) + "-");
                }

                //Auto-Regressive Posture
                for (int k = 0; k < current.ActorPosture.Length; k++)
                {
                    X.Feed(current.ActorPosture[k].GetPosition().PositionTo(current.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Position");
                    X.Feed(current.ActorPosture[k].GetForward().DirectionTo(current.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Forward");
                    X.Feed(current.ActorPosture[k].GetUp().DirectionTo(current.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Up");
                    X.Feed(current.ActorVelocities[k].DirectionTo(current.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Velocity");
                }

                //Gating Variables
                {
                    int index = 0;
                    for (int k = 0; k < current.TimeSeries.Samples.Length; k++)
                    {
                        for (int b = 0; b < current.PhaseSeries.Bones.Length; b++)
                        {
                            if (contacts.Contains(current.PhaseSeries.Bones[b]))
                            {
                                Vector2 phase = current.PhaseSeries.Amplitudes[k][b] * Utility.PhaseVector(current.PhaseSeries.Phases[k][b]);
                                index += 1;
                                X.Feed(phase.x, "Gating" + index + "-Key" + (k + 1) + "-Bone" + current.PhaseSeries.Bones[b]);
                                index += 1;
                                X.Feed(phase.y, "Gating" + index + "-Key" + (k + 1) + "-Bone" + current.PhaseSeries.Bones[b]);
                            }
                        }
                    }
                }

                //Output
                //Root Update
                Matrix4x4 delta = next.Root.TransformationTo(current.Root);
                Y.Feed(new Vector3(delta.GetPosition().x, Vector3.SignedAngle(Vector3.forward, delta.GetForward(), Vector3.up), delta.GetPosition().z), "RootUpdate");
                Y.FeedXZ(next.RootSeries.Velocities[next.TimeSeries.Pivot].DirectionTo(next.Root), "RootVelocity");
                Y.Feed(next.StyleSeries.GetValues(next.TimeSeries.Pivot, styles), "RootActions");

                //Control
                for (int k = next.TimeSeries.Pivot + 1; k < next.TimeSeries.Samples.Length; k++)
                {
                    Y.FeedXZ(next.RootSeries.GetPosition(k).PositionTo(next.Root), "TrajectoryPosition" + (k + 1));
                    Y.FeedXZ(next.RootSeries.GetDirection(k).DirectionTo(next.Root), "TrajectoryDirection" + (k + 1));
                    Y.FeedXZ(next.RootSeries.GetVelocity(k).DirectionTo(next.Root), "TrajectoryVelocity" + (k + 1));
                    Y.Feed(next.StyleSeries.GetValues(k, styles), "Actions" + (k + 1) + "-");
                }

                //Auto-Regressive Posture
                for (int k = 0; k < next.ActorPosture.Length; k++)
                {
                    Y.Feed(next.ActorPosture[k].GetPosition().PositionTo(next.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Position");
                    Y.Feed(next.ActorPosture[k].GetForward().DirectionTo(next.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Forward");
                    Y.Feed(next.ActorPosture[k].GetUp().DirectionTo(next.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Up");
                    Y.Feed(next.ActorVelocities[k].DirectionTo(next.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Velocity");
                }

                //Contacts
                Y.Feed(next.ContactSeries.GetContacts(next.TimeSeries.Pivot, contacts), "Contacts-");

                //Phase Update
                for (int k = next.TimeSeries.Pivot; k < next.TimeSeries.Samples.Length; k++)
                {
                    for (int b = 0; b < next.PhaseSeries.Bones.Length; b++)
                    {
                        if (contacts.Contains(next.PhaseSeries.Bones[b]))
                        {
                            Y.Feed(next.PhaseSeries.Amplitudes[k][b] * Utility.PhaseVector(Utility.SignedPhaseUpdate(current.PhaseSeries.Phases[k][b], next.PhaseSeries.Phases[k][b])), "PhaseUpdate-" + (k + 1) + "-" + (b + 1));
                            Y.Feed(next.PhaseSeries.Amplitudes[k][b] * Utility.PhaseVector(next.PhaseSeries.Phases[k][b]), "PhaseState-" + (k + 1) + "-" + (b + 1));
                        }
                    }
                }
            }

            private class Container
            {
                public MotionAsset Asset;
                public Frame Frame;
                public Actor Actor;

                public TimeSeries TimeSeries;
                public RootModule.Series RootSeries;
                public StyleModule.Series StyleSeries;
                public ContactModule.Series ContactSeries;
                public PhaseModule.Series PhaseSeries;

                //Actor Features
                public Matrix4x4 Root;
                public Matrix4x4[] ActorPosture;
                public Vector3[] ActorVelocities;

                public Container(DinoPipeline setup, float timestamp)
                {
                    MotionEditor editor = setup.Pipeline.GetEditor();
                    editor.LoadFrame(timestamp);
                    Asset = editor.GetSession().Asset;
                    Frame = editor.GetCurrentFrame();

                    TimeSeries = editor.GetTimeSeries();
                    RootSeries = Asset.GetModule<RootModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror) as RootModule.Series;
                    StyleSeries = Asset.GetModule<StyleModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror) as StyleModule.Series;
                    ContactSeries = Asset.GetModule<ContactModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror) as ContactModule.Series;
                    PhaseSeries = Asset.GetModule<PhaseModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror) as PhaseModule.Series;

                    Root = editor.GetSession().GetActor().transform.GetWorldMatrix();
                    ActorPosture = editor.GetSession().GetActor().GetBoneTransformations();
                    ActorVelocities = editor.GetSession().GetActor().GetBoneVelocities();
                }
            }
        }

        private class ControllerMANNSetup
        {
            public static void Export(DinoPipeline setup, AssetPipeline.Data X, AssetPipeline.Data Y, float tCurrent, float tNext)
            {
                Container current = new Container(setup, tCurrent);
                Container next = new Container(setup, tNext);

                string[] styles = new string[] { "Idle", "Move", "Speed" };
                // string[] gating = new string[]{"LeftHandSite", "RightHandSite", "LeftFootSite", "RightFootSite"};
                // string[] contacts = new string[]{"LeftHandSite", "RightHandSite", "LeftFootSite", "RightFootSite"};
                string[] gating = new string[] { "AnzB:LeftFootIndex4", "AnzB:RightFootIndex4" };
                string[] contacts = new string[] { "AnzB:LeftFootIndex4", "AnzB:RightFootIndex4" };

                //Input
                //Control
                for (int k = 0; k < current.TimeSeries.Samples.Length; k++)
                {
                    X.FeedXZ(next.RootSeries.GetPosition(k).PositionTo(current.Root), "TrajectoryPosition" + (k + 1));
                    X.FeedXZ(next.RootSeries.GetDirection(k).DirectionTo(current.Root), "TrajectoryDirection" + (k + 1));
                    X.FeedXZ(next.RootSeries.GetVelocity(k).DirectionTo(current.Root), "TrajectoryVelocity" + (k + 1));
                    X.Feed(next.StyleSeries.GetValues(k, styles), "Actions" + (k + 1) + "-");
                }

                //Auto-Regressive Posture
                for (int k = 0; k < current.ActorPosture.Length; k++)
                {
                    X.Feed(current.ActorPosture[k].GetPosition().PositionTo(current.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Position");
                    X.Feed(current.ActorPosture[k].GetForward().DirectionTo(current.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Forward");
                    X.Feed(current.ActorPosture[k].GetUp().DirectionTo(current.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Up");
                    X.Feed(current.ActorVelocities[k].DirectionTo(current.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Velocity");
                }

                //Gating Variables
                {
                    int[] indices = setup.Pipeline.GetEditor().GetSession().GetActor().GetBoneIndices(gating);
                    for (int i = 0; i < gating.Length; i++)
                    {
                        Vector3 velocity = current.ActorVelocities[indices[i]].DirectionTo(current.Root);
                        X.Feed(velocity, "GatingVelocity" + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[indices[i]].GetName());
                    }
                }

                //Output
                //Root Update
                Matrix4x4 delta = next.Root.TransformationTo(current.Root);
                Y.Feed(new Vector3(delta.GetPosition().x, Vector3.SignedAngle(Vector3.forward, delta.GetForward(), Vector3.up), delta.GetPosition().z), "RootUpdate");
                Y.FeedXZ(next.RootSeries.Velocities[next.TimeSeries.Pivot].DirectionTo(next.Root), "RootVelocity");
                Y.Feed(next.StyleSeries.GetValues(next.TimeSeries.Pivot, styles), "RootActions");

                //Control
                for (int k = next.TimeSeries.Pivot + 1; k < next.TimeSeries.Samples.Length; k++)
                {
                    Y.FeedXZ(next.RootSeries.GetPosition(k).PositionTo(next.Root), "TrajectoryPosition" + (k + 1));
                    Y.FeedXZ(next.RootSeries.GetDirection(k).DirectionTo(next.Root), "TrajectoryDirection" + (k + 1));
                    Y.FeedXZ(next.RootSeries.GetVelocity(k).DirectionTo(next.Root), "TrajectoryVelocity" + (k + 1));
                    Y.Feed(next.StyleSeries.GetValues(k, styles), "Actions" + (k + 1) + "-");
                }

                //Auto-Regressive Posture
                for (int k = 0; k < next.ActorPosture.Length; k++)
                {
                    Y.Feed(next.ActorPosture[k].GetPosition().PositionTo(next.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Position");
                    Y.Feed(next.ActorPosture[k].GetForward().DirectionTo(next.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Forward");
                    Y.Feed(next.ActorPosture[k].GetUp().DirectionTo(next.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Up");
                    Y.Feed(next.ActorVelocities[k].DirectionTo(next.Root), "Bone" + (k + 1) + setup.Pipeline.GetEditor().GetSession().GetActor().Bones[k].GetName() + "Velocity");
                }

                //Contacts
                Y.Feed(next.ContactSeries.GetContacts(next.TimeSeries.Pivot, contacts), "Contacts-");
            }

            private class Container
            {
                public MotionAsset Asset;
                public Frame Frame;
                public Actor Actor;

                public TimeSeries TimeSeries;
                public RootModule.Series RootSeries;
                public StyleModule.Series StyleSeries;
                public ContactModule.Series ContactSeries;

                //Actor Features
                public Matrix4x4 Root;
                public Matrix4x4[] ActorPosture;
                public Vector3[] ActorVelocities;

                public Container(DinoPipeline setup, float timestamp)
                {
                    MotionEditor editor = setup.Pipeline.GetEditor();
                    editor.LoadFrame(timestamp);
                    Asset = editor.GetSession().Asset;
                    Frame = editor.GetCurrentFrame();

                    TimeSeries = editor.GetTimeSeries();
                    RootSeries = Asset.GetModule<RootModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror) as RootModule.Series;
                    StyleSeries = Asset.GetModule<StyleModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror) as StyleModule.Series;
                    ContactSeries = Asset.GetModule<ContactModule>().ExtractSeries(TimeSeries, timestamp, editor.Mirror) as ContactModule.Series;

                    Root = editor.GetSession().GetActor().transform.GetWorldMatrix();
                    ActorPosture = editor.GetSession().GetActor().GetBoneTransformations();
                    ActorVelocities = editor.GetSession().GetActor().GetBoneVelocities();
                }
            }
        }

    }
}
#endif










// {
//     //TODO: replace like in soccer pipeline
//     StyleModule module = asset.HasModule<StyleModule>() ? asset.GetModule<StyleModule>() : asset.AddModule<StyleModule>();
//     RootModule root = asset.GetModule<RootModule>();
//     ContactModule contact = asset.GetModule<ContactModule>();
//     module.Clear();
//     StyleModule.StyleFunction idling = module.AddStyle("Idle");
//     StyleModule.StyleFunction moving = module.AddStyle("Move");
//     StyleModule.StyleFunction sitting = module.AddStyle("Sit");
//     StyleModule.StyleFunction resting = module.AddStyle("Rest");
//     StyleModule.StyleFunction standing = module.AddStyle("Stand");
//     StyleModule.StyleFunction jumping = module.AddStyle("Jump");
//     StyleModule.StyleFunction speed = module.AddStyle("Speed");
//     float[] timeWindow = asset.GetTimeWindow(Pipeline.GetEditor().PastWindow + Pipeline.GetEditor().FutureWindow, 1f);
//     float[] contactHeights = new float[asset.Frames.Length];
//     for(int i=0; i<asset.Frames.Length; i++) {
//         for(int j=0; j<contact.Sensors.Length; j++) {
//             contactHeights[i] += asset.Frames[i].GetBoneTransformation(contact.Sensors[j].Bone, false).GetPosition().y;
//         }
//         contactHeights[i] /= contact.Sensors.Length;
//     }
//     for(int f=0; f<asset.Frames.Length; f++) {
//         float weight = GetMovementWeight(asset.Frames[f].Timestamp, 0.5f);
//         idling.Values[f] = 1f - weight;
//         moving.Values[f] = weight;
//         float sit = GetContactsWeight(asset.Frames[f].Timestamp, 0.5f, contact, sitPatterns, 0f, 1f);
//         float rest = GetContactsWeight(asset.Frames[f].Timestamp, 0.5f, contact, restPatterns, 0f, 1f);
//         float stand = GetContactsWeight(asset.Frames[f].Timestamp, 0.5f, contact, standPatterns, 0f, 1f);
//         float jump = GetContactsWeight(asset.Frames[f].Timestamp, 0.5f, contact, jumpPatterns, 0.3f, 0.1f);
//         float[] actions = new float[]{sit, rest, stand, jump};
//         Utility.SoftMax(ref actions);
//         sitting.Values[f] = sit;
//         resting.Values[f] = rest;
//         standing.Values[f] = stand;
//         jumping.Values[f] = jump;
//         speed.Values[f] = root.GetRootLength(asset.Frames[f].Timestamp, false); //TODO: Divide by root window
//     }

//     float GetMovementWeight(float timestamp, float threshold) {
//         float[] weights = new float[timeWindow.Length];
//         for(int j=0; j<timeWindow.Length; j++) {
//             weights[j] = Mathf.Max(
//                             root.GetRootVelocity(timestamp + timeWindow[j], false).magnitude,
//                             root.GetRootAngle(timestamp + timeWindow[j], false)
//                         ).Ratio(0f, threshold);
//         }

//         float[] gradients = new float[weights.Length-1];
//         for(int i=0; i<gradients.Length; i++) {
//             gradients[i] = (weights[i+1] - weights[i]) / (timeWindow[i+1] - timeWindow[i]);
//         }

//         return weights.Gaussian(Mathf.Abs(gradients.Gaussian())).SmoothStep(2f, 0.5f);
//     }

//     float GetContactsWeight(float timestamp, float window, ContactModule module, List<float[]> patterns, float heightThreshold, float power) {
//         float ContactGaussian(float t) {
//             float[] weights = new float[timeWindow.Length];
//             for(int j=0; j<timeWindow.Length; j++) {
//                 bool match = false;
//                 for(int i=0; i<patterns.Count; i++) {
//                     float[] contacts = module.GetContacts(t + timeWindow[j], false);
//                     match = ArrayExtensions.Equal(contacts, patterns[i]).All(true);
//                     if(match) {
//                         break;
//                     }
//                 }
//                 if(match && heightThreshold != 0f && contactHeights[asset.GetFrame(t).Index-1] < heightThreshold) {
//                     match = false;
//                 }
//                 weights[j] = match ? 1f : 0f;
//             }
//             return weights.Gaussian();
//         }
//         float weight = ContactGaussian(timestamp);
//         weight = Mathf.Pow(weight, 1f-weight);
//         return Mathf.Pow(weight, power);
//     }
// }