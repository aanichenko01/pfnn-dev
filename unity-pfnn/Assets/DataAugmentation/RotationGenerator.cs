using UnityEngine;
using UnityEditor;
using UnityEditor.Animations;
using System.Collections;
using System;

public class RotationGenerator : MonoBehaviour
{

    Animator Animator;
    GameObjectRecorder Recorder;
    [SerializeField] int NumberOfRotationsToGenerate = 10;
    [SerializeField] string SaveFolder = "Assets/";
    [SerializeField] float Framerate = 30;
    [SerializeField] string FileName = null;

    private AnimationClip Clip;
    private AnimationClip CurrentClip;
    private string CurrentClipName;
    private bool CanRecord;

    // booleans for tracking blendtree loop
    private int currentTime = -1;
    private int previousTime = -1;

    // Start is called before the first frame update
    void Start()
    {
        Animator = GetComponent<Animator>();
        StartCoroutine(GenerateAndRecordRotations());
    }

    IEnumerator GenerateAndRecordRotations()
    {

        int total = 0;
        int previousTime = -1;

        float increment = 360f / NumberOfRotationsToGenerate;

        // Loop over combinations of height and distance to generate augmented animations
        for (int i = 1; i <= NumberOfRotationsToGenerate; i++)
        {
            // Rotate gameobject
            transform.rotation = Quaternion.Euler(0, i * increment, 0);
            // Reset recorder for each clip (otherwise clips get appended)
            Recorder = new GameObjectRecorder(gameObject);
            Recorder.BindComponentsOfType<Transform>(gameObject, true);

            StartRecording(total);
            yield return new WaitUntil(() => isAnimationDone());
            StopRecording();
            previousTime = currentTime;

            total += 1;

        }

        Debug.Log($"Generated {total} rotated variations");
    }

    public bool isAnimationDone()
    {
        currentTime = (int)Math.Floor(Animator.GetCurrentAnimatorStateInfo(0).normalizedTime);
        if (currentTime > previousTime)
        {
            previousTime = currentTime;
            return true;
        }
        else
        {
            return false;
        }
    }
    private void StartRecording(int index)
    {
        CanRecord = true;
        Clip = new AnimationClip
        {
            frameRate = Framerate,
            name = $"{FileName}_{index}"
        };
        CurrentClipName = Clip.name;
        CurrentClip = Clip;
    }

    private void StopRecording()
    {
        CanRecord = false;
        Recorder.SaveToClip(CurrentClip);
        AssetDatabase.CreateAsset(CurrentClip, SaveFolder + CurrentClipName + ".anim");
        AssetDatabase.SaveAssets();
    }

    private void LateUpdate()
    {
        if (Clip == null) return;

        if (CanRecord)
        {
            Recorder.TakeSnapshot(Time.deltaTime);
        }
    }

}
