using UnityEngine;
using UnityEditor;
using UnityEditor.Animations;
using System.Collections;
using System;

public class IdleGenerator : MonoBehaviour
{

    Animator Animator;
    GameObjectRecorder Recorder;
    
    [SerializeField] float Increment = 0.5f;
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
        StartCoroutine(GenerateAndRecordJumps());
    }

    IEnumerator GenerateAndRecordJumps()
    {

        int total = 0;
        int previousTime = -1;

        // Loop over combinations of height and distance to generate augmented animations
        // Anything more than 0.3 is a bit too much
        for (float blend = 0f; blend <= 0.5f; blend += Increment)
        {

                // Animator.SetFloat("Blend", blend);
                Animator.SetLayerWeight(2, blend);

                // Reset recorder for each clip (otherwise clips get appended)
                Recorder = new GameObjectRecorder(gameObject);
                Recorder.BindComponentsOfType<Transform>(gameObject, true);

                StartRecording(total);
                yield return new WaitUntil(() => isAnimationDone());
                StopRecording();
                previousTime = currentTime;
                Debug.Log($"Recorded for blend: {blend}");

                total += 1;
        }

        // Reset back to idle state
        Animator.SetLayerWeight(2, 0);
        Debug.Log($"Generated {total} idle variations");
    }

    public bool isAnimationDone() 
    {
        currentTime = (int)Math.Floor(Animator.GetCurrentAnimatorStateInfo(2).normalizedTime);
        if(currentTime > previousTime) {
            previousTime = currentTime;
            return true;
        } else {
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
