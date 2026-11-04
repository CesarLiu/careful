# Right-of-way conflicts: record the facts (do not decide "aggressive")

Each row of `conflicts.csv` is a scene where the two objects of interest (the self-driving car SDC, blue, and its
partner OOI, orange) swept the same patch of road: one **left it** (`first`, at `first_left_t`) and the other
**entered it** `pet` seconds later (`second_in_t`). The red band in the figure's time panels is that interval.
Lane changes and following are handled elsewhere; here only crossings and merges.

You do **not** say whether anyone was aggressive. You record **facts about the right of way and the reaction**, from
the figure (`logs/responsibility/ooi_labels_v2/figures/<scene>.png`) and the CSV row. A script derives the label from
your facts, so be literal and consistent. Look at every figure. If it does not settle who had priority or why the
second car slowed, load the scene data:

```
cd /home/Cesar/catspond-cr; ~/venvs/cat39/bin/python
from responsibility.scene import Scene; s = Scene.load("raw_scenes_500/<scene>.pkl")
# s.sdc, s.objects_of_interest, s.position[i,t,:2], s.velocity[i,t], s.heading[i,t], s.valid[i,t]; 10 Hz, 91 steps;
# s.map_features, s.dynamic_map_states (traffic lights: state per step; stop_point)
```

## What the figure shows

Left: the map at the moment the first car left the shared area. Both tracks (a dot every second, labelled with the
time), the two boxes, other agents grey, traffic-light stop points coloured by their state at that step (green /
yellow / red; grey unknown), stop signs as dark-red octagons, the star the first crossing of the two tracks.
Right, top to bottom: each car's distance past the conflict point; both speeds; the gap and the time headways; and
two bars, the **control** each car faced: the light's colour over time (a black mark under the bar: the car stood
still, < 0.3 m/s), or a pink bar for a stop sign; a coloured tick where it crossed its stop line. The caption gives
the rule-based reading of the right of way (`rule`, `holder`); **it is often wrong** (roundabouts, the first car at an
all-way stop, left turns read the wrong way round, signals of the wrong approach, "signal-unknown"). Judge it yourself.

## What to record, per conflict

- `holder`: who had the right of way over the other **at the moment the second car had to decide**: `sdc`, `ooi`, `none`
  (equal claims, no rule applies) or `unclear` (cannot tell). Use the California Vehicle Code in this order:
  1. a signal: green or green arrow beats red; at a green ball a **left-turner yields to oncoming** traffic (§21801);
     a right turn on red yields to the cross traffic (§21453b);
  2. a stop or yield sign on one approach only: the sign-controlled car yields to the cross traffic (§21802, 21803);
  3. an all-way stop: whoever stopped first goes first; on a tie the car on the right (§21800c);
  4. no control: the car already in the intersection, then the one on the right, a terminating road yields to the
     continuing one (§21800);
  5. a driveway or alley yields to the road (§21804); entering a roundabout yields to circulating cars;
  6. a car that lawfully yielded and entered becomes the holder (§21801b, 21802b): if the first car had waited its
     turn and was already moving when the other arrived, it holds the right of way.
- `basis`: one of `light`, `stop_sign`, `all_way_arrival`, `left_turn`, `right_on_red`, `uncontrolled`, `driveway`,
  `roundabout`, `lawful_entry`, `merge`, `other`, `none`, then a few words (e.g. "light: SDC green, OOI red").
- `first_violated_control`: `yes` if the **first** car broke its own control in this conflict: entered the conflict on
  a red, or passed a stop sign without stopping (the CSV gives `first_min_speed_before_line`: below about 0.5 m/s
  it stopped, above about 2.5 m/s it rolled through; between, judge from the speed curve), or went before its turn
  at an all-way stop; `no` otherwise; `na` when it faced no control.
- `reaction`: how the **second** car's speed changed between about 3 s before it entered and its entry
  (`second_speed_drop`, `second_peak_decel` in the CSV; an isolated one-frame spike in the speed curve is noise, not
  braking): `none` (under about 1 m/s change), `slowed` (a real slowdown), `hard` (peak decel at least 3.4 m/s^2 in
  more than one frame), `stopped` (forced to stop from at least 5 m/s).
- `reaction_cause`: for `slowed`, `hard` or `stopped`, why the second car slowed: `first_car` (it slowed because the
  first car went ahead of it: its deceleration began after the first car committed to the conflict, and no other
  cause explains it), `own_turn` (it was turning), `red_or_stop` (its own light or sign), `lead_vehicle`, `other`,
  or `none` when `reaction` is `none`.
- `confidence`: `high`, `medium` or `low`.
- `note`: one sentence with the deciding numbers and times.

## Output

`scene,holder,basis,first_violated_control,reaction,reaction_cause,confidence,note` (quote the text fields), one row per
conflict of your list, to the file named in your task. Write it incrementally. Do not read anyone else's output.
