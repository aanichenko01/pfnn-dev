using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Waypoints : MonoBehaviour
{

    [SerializeField] private float waypointSize = 0.25f;

    // method that is called in the editor so we can draw something in scene view
    // ONDRAWGIZMOS ONLY CALLED IN SCENE VIEW so just for visualization
    private void OnDrawGizmos()
    {
        // takes every transform that is a child of our current waypoint system
        foreach (Transform t in transform)
        {
            Gizmos.color = Color.blue;
            Gizmos.DrawWireSphere(t.position, waypointSize);
        }

        Gizmos.color = Color.red;
        for (int i = 0; i < transform.childCount - 1; i++)
        {
            Gizmos.DrawLine(transform.GetChild(i).position, transform.GetChild(i + 1).position);
        }

        // Close the loop from last waypoint to first
        Gizmos.DrawLine(transform.GetChild(transform.childCount - 1).position, transform.GetChild(0).position);

    }

    public Transform GetNextWaypoint(Transform currentWaypoint)
    {
        if (currentWaypoint == null)
        {
            return transform.GetChild(0);
        }

        // get sibling gives us the specific index of the current object
        if (currentWaypoint.GetSiblingIndex() < transform.childCount-1)
        {
            return transform.GetChild(currentWaypoint.GetSiblingIndex() + 1);
        } else {
            // Only needed for looping trajectory (don't think I need it for jump)
            return transform.GetChild(0);
        }
    }

}
