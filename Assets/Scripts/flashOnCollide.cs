using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class flashOnCollide : MonoBehaviour
{
    public Renderer rend = null;
    public int speed = 1;
    Color startColor;
    [SerializeField] //to see the variable in the editor but keep the variable perivate
    bool hasCollided = false;

    private void Awake() 
    {
        //here you could load the data
        rend = GetComponentInChildren<Renderer>();
    }
    // Start is called before the first frame update
    void Start()
    {
        startColor = rend.material.color;
    }

    // Update is called once per frame
    void Update()
    {
        if (hasCollided == false)
            return;

        var currentColor = rend.material.color;

        if (currentColor == startColor)
        {
            return;

        }
        var targetColor = Color.Lerp(currentColor, startColor, Time.deltaTime * speed); // after collinding the color will go back to the starting color
        rend.material.color = targetColor;
        hasCollided = targetColor != currentColor;
    }


    private void OnCollisionEnter(Collision collision)
    {
        hasCollided = true;
        Color newColor = Color.HSVToRGB(Random.value, 1.0f, 1.0f);
        rend.material.color = newColor;
        Debug.Log($"Collided with {collision.gameObject.name}");

    }
}

