using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WaypointController : MonoBehaviour
{

    // Stores reference to waypoints
    [SerializeField] private Waypoints waypoints;

    [SerializeField] private float speed = 5f;

    [SerializeField] private float threshold = 0.1f;

    private Transform currentWaypoint;

    // Start is called before the first frame update
    void Start()
    {

        // Move player to first waypoint
        currentWaypoint = waypoints.GetNextWaypoint(currentWaypoint);
        transform.position = currentWaypoint.position;

        //Set next waypoint target
        currentWaypoint = waypoints.GetNextWaypoint(currentWaypoint);
        transform.LookAt(currentWaypoint);

    }

    // Update is called once per frame
    void Update()
    {

        transform.position = Vector3.MoveTowards(transform.position, currentWaypoint.position, speed * Time.deltaTime);
        // If close enough to waypoint get the next one
        if (Vector3.Distance(transform.position, currentWaypoint.position) < threshold)
        {
            currentWaypoint = waypoints.GetNextWaypoint(currentWaypoint);
            transform.LookAt(currentWaypoint);
        }

    }
}
