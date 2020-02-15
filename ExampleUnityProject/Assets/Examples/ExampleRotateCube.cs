using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace NvPipeUnity {

    public class ExampleRotateCube : MonoBehaviour {

        // Update is called once per frame
        void Update() {
            transform.Rotate(Vector3.right, Time.deltaTime * 360.0f);
        }
    }
}