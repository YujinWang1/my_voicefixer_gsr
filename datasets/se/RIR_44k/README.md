# RIR-44K Simulated Room Impulse Response Dataset

## Example

See the **example** folder.

## Naming Of The RIRs

```
rir_<pick_up_pattern>_rt60_<rt60>_room_<room_height_width_length>_mic_<mic_position>_source_<source_position>
```

- pick_up_pattern: cardioid or omnidirectional.
- rt60: [0.05, 1.0]
- room_height_width_length, mic_position, and source_position are three dimensional.

## The Simulation Process

We randomly simulated a collection of Room Impulse Response filters to simulate the 44.1kHz speech room reverberation using a open source tool (https://github.com/sunits/rir\_simulator\_python). The meters of height, width and length of the room is sampled randomly in a uniform distribution U(1,12). The placement of the microphone is then randomly selected within the room space. For the placement of sound source, we first determined the distance between the microphone and sound source, which is randomly sampled in a Gaussian distribution N(\mu,\sigma^2), \mu=2, \sigma=4. If the sampled value is negative or greater than five meters, we will sample the distance again until it meets the requirement. After determined the distance between the microphone and sound source, the placement of the sound source is randomly selected on the sphere centered at the microphone. The RT60 value we choose come from the uniform distribution  U(0.05,1.0). For the pickup pattern of the microphone, we randomly choose from types omnidirectional and cardioid. Finally, we simulated 43239 filters following this scheme, in which we randomly split out 5000 filters as test set \textit{*RIR-Test*} and named other 38239 filters as \textit{*RIR-Train*}.

## Author

HaoheLiu (haoheliu@gmail.com)