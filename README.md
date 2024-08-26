# Realtime-RiskNET

Original dev. code by Tim Schoonbeek. <br>

Real-time Collision Risk Prediction model, inferring on live camera feed. Models are optimized with TensorRT.

### Command to infer
```bash
python /path/to/RiskNET/main.py --run_name saved_run_name
```

"--camera" is initialized along with the above command to infer on live camera feed. Else, infers on a saved video from the videos folder. <br>
(Note: Might look slower when compared to processing a saved video. Saved videos are **_SAVED_**, the IDE doesn't have to wait for the next frame to arrive) <br>
"--viz" is initialized if you want to visualize predictions and processing in realtime. (Note: Makes the pipeline run slower) <br>
"--graph" is initialized if you want to see a real time graph for predictions, seq. time and prediction time. (Note: Makes the pipeline run slower) <br>
"--tenfps" is initialized to drop down from the original camera framerate to 10 fps and infer on it.
