using UnityEngine;
using UnityEditor;
using UnityEditor.Animations;
using System.Collections;
using System;

public class JumpGenerator : MonoBehaviour
{

    Animator Animator;
    GameObjectRecorder Recorder;
    
    [SerializeField] float Increment = 0.5f;
    [SerializeField] string SaveFolder = "Assets/";
    [SerializeField] float Framerate = 30;

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
        for (float distance = 0f; distance <= 1.1f; distance += Increment)
        {
            for (float height = 0f; height <= 1.1f; height += Increment)
            {

                Animator.SetFloat("Distance", distance);
                Animator.SetFloat("Height", height);

                // Reset recorder for each clip (otherwise clips get appended)
                Recorder = new GameObjectRecorder(gameObject);
                Recorder.BindComponentsOfType<Transform>(gameObject, true);

                StartRecording(total);
                yield return new WaitUntil(() => isAnimationDone());
                StopRecording();
                previousTime = currentTime;

                total += 1;
            }
        }

        // Reset back to idle state
        Animator.SetFloat("Height", 0);
        Animator.SetFloat("Distance", 0);
        Debug.Log($"Generated {total} jump variations");
    }

    public bool isAnimationDone() 
    {
        currentTime = (int)Math.Floor(Animator.GetCurrentAnimatorStateInfo(0).normalizedTime);
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
            name = $"{gameObject.name}_animation_{index}"
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
