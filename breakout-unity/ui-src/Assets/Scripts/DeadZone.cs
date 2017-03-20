using UnityEngine;
using System.Collections;

public class DeadZone : MonoBehaviour {

	void OnTriggerEnter(Collider col){
		GM.instance.loseLife ();
		Destroy(col.gameObject);
	}
}
