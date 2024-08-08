using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

public class SequenceMaker : MonoBehaviour
{

    Animator Animator;
    GameObjectRecorder Recorder;

    [SerializeField] string SaveFolder = "Assets/";
    [SerializeField] float Framerate = 30;
    [SerializeField] string FileName = null; 

    private AnimationClip Clip;
    private AnimationClip CurrentClip;
    private string CurrentClipName;
    private bool CanRecord;

    // var for tracking animations
    private AnimatorStateInfo currentStateInfo;

    // Start is called before the first frame update
    void Start()
    {
        
        Animator = GetComponent<Animator>();
        StartCoroutine(RecordSequence());
        
    }

IEnumerator RecordSequence()
    {

        // Reset recorder for each clip (otherwise clips get appended)
        Recorder = new GameObjectRecorder(gameObject);
        Recorder.BindComponentsOfType<Transform>(gameObject, true);

        StartRecording();
        yield return new WaitUntil(() => isAnimationDone());
        StopRecording();


    }

    public bool isAnimationDone() 
    {
        currentStateInfo = Animator.GetCurrentAnimatorStateInfo(0);

        // Check we are in walk animation
        if(currentStateInfo.IsName("Walk")) {
            // check walk animation has finished executing
            if(currentStateInfo.normalizedTime >= 1.0f) {
                return true;
            }
        }
        return false;
        
    }

    private void StartRecording()
    {
        CanRecord = true;
        Clip = new AnimationClip
        {
            frameRate = Framerate,
            name = FileName
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

    private void Update() {

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
