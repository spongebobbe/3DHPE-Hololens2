using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Utilities;
using Microsoft.MixedReality.Toolkit.Input;

public class HandsManager : MonoBehaviour, IMixedRealitySourceStateHandler
{
    IMixedRealityPointer pointer = null;
    uint sourceID = 0;
    public GameObject marker;
    void Update()
    {
        if (pointer == null)
            return;

        if (pointer.Result == null)
            return;

         
        Debug.Log($"Update pointer: {pointer}");
        Debug.Log($"Update marker: {marker.transform.position}");
        Debug.Log($"Update pointer.Result.Details.Point: {pointer.Result.Details.Point}");

        marker.transform.position = pointer.Result.Details.Point; //set the marker position equal to the position you are pointing at

    }
    public void OnSourceDetected(SourceStateEventData eventData)
    {
        var hand = eventData.Controller as IMixedRealityHand; // for example i am only interested in the right hand. if the casting is succesfull it means that the event is indeed a heand and var heand is not null
        if (hand == null || hand.ControllerHandedness != Handedness.Right )
            return;

        pointer = hand.InputSource.Pointers[0];
        Debug.Log($"OnSourceDetected pointer soruce detected: {pointer}");
        sourceID = eventData.SourceId;
        Debug.Log($"OnSourceDetected A source was detected: {eventData.InputSource.SourceName}");
    }

    public void OnSourceLost(SourceStateEventData eventData)
    {
        if (eventData.SourceId != sourceID) //if it's not the right heand I don't care and return
            return;

        pointer = null;
        sourceID = 0;
        Debug.Log($"A source was lost: {eventData.InputSource.SourceName}");

    }

    private void OnEnable()
    {
        CoreServices.InputSystem?.RegisterHandler<IMixedRealitySourceStateHandler>(this); //tells mixed reality to send here the events detected
    }

    private void OnDisable()
    {
        Debug.Log($"this is: {this.ToString()}");
        CoreServices.InputSystem?.UnregisterHandler<IMixedRealitySourceStateHandler>(this);
    }
}

