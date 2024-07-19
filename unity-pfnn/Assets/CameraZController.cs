using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraZController : MonoBehaviour
{
    public Transform player;
    public float smoothSpeed = 10f;

    private float playerZ;
    private Vector3 follow;
    private Vector3 offset;

    // Use this for initialization
    void Start () {
        // Optionally, set an initial offset if desired
        offset = new Vector3(1.325357f, 0.461543f, 0f);
    }

    // Update is called once per frame
    void LateUpdate () {
        playerZ = player.position.z;
        follow = new Vector3(offset.x, offset.y, playerZ);

        // Use Lerp to smooth the transition
        Vector3 smoothedPosition = Vector3.Lerp(transform.position, follow, smoothSpeed);
        transform.position = smoothedPosition;
    }
}
