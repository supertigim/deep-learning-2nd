using UnityEngine;
using System.Collections;

public class TimedDestroy : MonoBehaviour {

	public float destroyTime = 1f;

	// Use this for initialization
	void Start () {

		Destroy (gameObject, destroyTime);

	}
}
