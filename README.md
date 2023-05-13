# impossibly-good
Code for Impossibly Good Experts and How to Follow Them (ICLR 2023)
```
@inproceedings{walsmanimpossibly,
  title={Impossibly Good Experts and How to Follow Them},
  author={Walsman, Aaron and Zhang, Muru and Choudhury, Sanjiban and Fox, Dieter and Farhadi, Ali},
  booktitle={International Conference on Learning Representations}
}
```

The code here is pretty raw at the moment, with branches for the state after the initial submission, and after the additional rebuttal experiments.

This code was developed using [rl-starter-files](https://github.com/lcswillems/rl-starter-files) and [torch-ac](https://github.com/lcswillems/torch-ac) as a starting point.

# vizdoom experiments
The [vizdoom](https://github.com/Farama-Foundation/ViZDoom) maps were made in [slade](https://slade.mancubus.net/).  The levels can be found in the `vizdoom_levels` directory, where you will find:

```
monster_hall_a.cfg
monster_hall_b.cfg
monster_hall_a.wad
monster_hall_b.wad

monster_room_a.cfg
monster_room_b.cfg
monster_room_a.wad
monster_room_b.wad
```

These files should be copied to the `scenarios` directory under your `vizdoom` installation.  The a/b versions are for the the different monster spawn locations.  We found it faster and easier to build separate levels for the two different monster spawn locations and switch between them randomly inside our own gym wrapper than to figure out how to get vizdoom to randomly spawn the monsters.  This was a decision made for haste purposes rather than for a specific engineering reason.
