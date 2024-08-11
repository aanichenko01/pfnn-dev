using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEditor.Animations;
using UnityEngine;

public class AutoSequenceMaker : MonoBehaviour
{

    Animator Animator;
    GameObjectRecorder Recorder;

    [SerializeField] string SaveFolder = "Assets/";
    [SerializeField] float Framerate = 30;
    [SerializeField] string FileName = null;
    [SerializeField] AnimationClip StartClip;

    [SerializeField, Tooltip("Assets/Resources/")]
    // Note this path is top level Assets/Resources/AnimaClipToAppendFolder
    string AnimClipsToAppendFolder = "AnimClips/";

    private AnimationClip Clip;
    private AnimationClip CurrentClip;
    private string CurrentClipName;

    private string ClipToAppendName;
    private bool CanRecord;

    private AnimatorStateInfo currentStateInfo;

    // List to store clips to append to StartClip
    private AnimationClip[] ClipsToAppend;

    private int NumClipsGenerated = 0;

    // Start is called before the first frame update
    void Start()
    {
        Animator = GetComponent<Animator>();
        LoadAnimClips();
        StartCoroutine(RecordSequences());
    }

    private void LoadAnimClips()
    {

        ClipsToAppend = Resources.LoadAll<AnimationClip>(AnimClipsToAppendFolder);

        if (StartClip == null)
        {
            Debug.LogError("Please provice a start animation clip.");
        }

        if (ClipsToAppend.Length == 0)
        {
            Debug.LogError("Animation clips to append could not be found.");
        }
        else
        {
            Debug.Log($"Loaded {ClipsToAppend.Length} animation clips");
        }
    }

    IEnumerator RecordSequences()
    {

        foreach (var clip in ClipsToAppend)
        {
            // Reset recorder for each clip (otherwise clips get appended)
            Recorder = new GameObjectRecorder(gameObject);
            Recorder.BindComponentsOfType<Transform>(gameObject, true);

            StartRecording();
            Animator.Play(StartClip.name);
            yield return new WaitUntil(() => isStartAnimationDone());

            ClipToAppendName = clip.name;
            Animator.Play(ClipToAppendName);
            yield return new WaitUntil(() => isAppendAnimationDone());
            StopRecording();

            NumClipsGenerated += 1;
            Debug.Log($"Saved clip {NumClipsGenerated}");
        }

        Debug.Log($"Generated {NumClipsGenerated} seqeunces");

    }

    public bool isStartAnimationDone()
    {
        currentStateInfo = Animator.GetCurrentAnimatorStateInfo(0);

        // Check start animation is playing
        if (currentStateInfo.IsName(StartClip.name))
        {
            // Check for when it finished executing
            if (currentStateInfo.normalizedTime >= 1.0f)
            {
                return true;
            }
        }
        return false;

    }

    public bool isAppendAnimationDone()
    {
        currentStateInfo = Animator.GetCurrentAnimatorStateInfo(0);

        // Check append animation is playing
        if (currentStateInfo.IsName(ClipToAppendName))
        {
            // Check for when it finished executing
            if (currentStateInfo.normalizedTime >= 1.0f)
            {
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
        AssetDatabase.CreateAsset(CurrentClip, $"{SaveFolder}{CurrentClipName}_{NumClipsGenerated}.anim");
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
