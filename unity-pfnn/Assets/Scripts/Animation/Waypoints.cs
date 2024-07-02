using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Waypoints : MonoBehaviour
{

    [SerializeField] private float WaypointSize = 0.1f;

    // Drawing only visible in scene view
    private void OnDrawGizmos()
    {
        // Loop over waypoint objects and draw sphere
        foreach (Transform t in transform)
        {
            Gizmos.color = Color.blue;
            Gizmos.DrawWireSphere(t.position, WaypointSize);
        }

        // Connect with lines
        Gizmos.color = Color.red;
        for (int i = 0; i < transform.childCount - 1; i++)
        {
            Gizmos.DrawLine(transform.GetChild(i).position, transform.GetChild(i + 1).position);
        }

        // Close the loop from last waypoint to first
        // Gizmos.DrawLine(transform.GetChild(transform.childCount - 1).position, transform.GetChild(0).position);

    }

    public Transform GetNextWaypoint(Transform currentWaypoint)
    {
        // If no waypoint initialized return first
        if (currentWaypoint == null)
        {
            return transform.GetChild(0);
        }

        // Loop to next waypoint
        if (currentWaypoint.GetSiblingIndex() < transform.childCount - 1)
        {
            return transform.GetChild(currentWaypoint.GetSiblingIndex() + 1);
        }
        // else
        // {
        //     // Loop back to first waypoint
        //     return transform.GetChild(0);
        // }
        else
        {
            // Return last waypoint so there is no loop
            return currentWaypoint;
        }
    }

    public Transform GetPreviousWaypoint(Transform currentWaypoint)
    {
        // If no waypoint initialized return first
        if (currentWaypoint == null)
        {
            return transform.GetChild(0);
        }

        // Loop to prev waypoint
        if (currentWaypoint.GetSiblingIndex() > 0)
        {
            return transform.GetChild(currentWaypoint.GetSiblingIndex() - 1);
        }
        else
        {
            // Loop back to last waypoint
            return transform.GetChild(transform.childCount - 1);
        }
    }

}
