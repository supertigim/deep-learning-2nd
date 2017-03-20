using UnityEngine;
using System.Collections;

public class Paddle : MonoBehaviour {

	public float paddleSpeed = 1.0f;
	private Vector3 PlayerPos = new Vector3 (0, -0.95f, 0);

	// Update is called once per frame
	void Update () {
		float xPos = transform.position.x + (Input.GetAxis("Horizontal") * paddleSpeed);
		PlayerPos = new Vector3 (Mathf.Clamp(xPos, -8f, 8f), -9.5f, 0f);

		transform.position = PlayerPos;
	}
}
