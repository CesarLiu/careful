# Aggressive or not: the two objects of interest of raw_scenes_500

`raw_scenes_500_ooi_aggression.csv` labels each of CAT's 500 logged scenes
(`raw_scenes_500/<scene>.pkl`) twice: is the self-driving car (SDC)
aggressive toward its partner object of interest (OOI), and is the OOI
aggressive toward the SDC. Each label covers that pair only; neither car's
behaviour toward third vehicles counts. The labels are ex post, over the
whole 9.1 s clip, and binary.

There are two passes. The lane pass (cut-ins, tailgating, close passes)
is rule-checkable. The right-of-way pass is what needs judgment: who had to
yield, and did going first force the other to. A first version labelled
only the lane pass and thereby only found cut-ins; this version adds the
right-of-way pass, with a standard taken from the literature.

## The standard

X is aggressive toward Y when X's own action shifted risk or effort onto Y
beyond what the situation required. Holding the right of way does not
excuse a compressed safety margin.

NHTSA's, Tasca's (2000) and the AAA Foundation's definitions all count
"failure to yield" and forcing others to yield as aggressive. Traffic
psychology draws no operational line between assertive and aggressive
driving. Traffic engineering treats short gap taking as normal: in "limited
priority" merging the major stream slows a little to let a minor-stream
vehicle in (Troutbeck and Kako 1999), and RSS grants permission to be
assertive within limits (Shalev-Shwartz et al. 2017; Shea-Blymyer and Abbas
2021, who call the line "fundamentally social"). Each yield duty in the
California Vehicle Code carries a proximity clause ("close enough to
constitute an immediate hazard", §21801, 21802, 21803, 21804): going first
is a violation only when the holder had to depart from its unimpeded
motion. So the line is drawn by what the other was made to do.

### Lane pass

| kind | X is aggressive when |
|---|---|
| cut in | X moved onto Y's path ahead of Y and left Y a time headway below 0.9 s, or made it brake harder than 2 m/s². This counts when a faster X pulls away after. |
| tailgated | X followed Y in its lane at ≥ 5 m/s with a headway below 0.9 s for 1 s or more, or closed in with a time to collision below 2 s. 0.9 s is German practice. The first 3 s after Y cut in ahead do not count unless X was closing. |
| close pass | The boxes came within 0.5 m at more than 3 m/s relative speed, and X caused it (closed in sideways, or went second and squeezed behind Y). |

### Right-of-way pass

For every pair whose two swept footprints overlap at a crossing or merge
(257 of 500 scenes), the conflict is the stretch of road both swept: one
car, F, left it, and the other, S, entered it `PET` seconds later (post-encroachment
time on the swept boxes, as in SSAM). The annotators record facts only (guide:
`row_of_way_guide.md`):

- who held the right of way, under the California Vehicle Code (§21453
  lights, §21801 left turns, §21802/21803 stop and yield signs, §21800
  all-way stops by arrival and uncontrolled junctions, §21804 driveways, and a
  car that lawfully yielded and entered becoming the holder), and the basis;
- whether F broke its own control (entered on red; passed a stop sign
  without stopping; went out of turn at an all-way stop);
- why S slowed, if it did: because of F, its own turn, its own light or
  sign, a lead vehicle.

The size of S's reaction is measured, not read from a figure
(`conflicts.csv`): in the window from 1 s before F entered the shared area
to S entering it, on the speed filtered over 0.5 s (so one-frame tracking
spikes drop out), `slowed` is a fall of at least 1 m/s from the running
peak, `hard` a peak braking of at least 3.4 m/s², `stopped` a stop from at
least 5 m/s. A reaction before F got there is the second car yielding in
good time, not being forced. `derive_row_labels.py` then draws the verdict
on F, the car that went first:

| F is aggressive when | thresholds |
|---|---|
| it went first without the right of way (S held it, or nobody) and the conflict was close | PET below 1.5 s at a crossing (Hyden's serious conflict; SSAM); below 0.9 s at a merge (paths less than 45° apart, which run on together). Not for two cars passing each other (paths 165° or more apart, or tracks that never cross). Or S's reaction was `hard` or `stopped` because of F. |
| the same, in the middle band, and F also broke its own control | PET up to 2 s, or S slowed because of F; at a crossing, or at a red light at a merge. A stop-sign violation counts only with 3 s of record before the line, and for a car that slowed on the way |
| it held the right of way and still forced S to brake `hard` to a near stop (below 2 m/s) | PET below 2 s: holding it does not excuse a margin that cost the other that much. |

Where the holder is unclear, F is aggressive only if it forced a hard
reaction. The holder is taken as unclear when it was read from a stop sign
that F passed at 5 m/s or more without ever slowing (a sign matched to the
wrong approach), or from the rule that a left turn yields to oncoming
traffic while F's own light is not in the map (a protected left turn cannot
be ruled out). Otherwise not aggressive: a
gap taken with S slowing only a little is assertive at most. S is never
aggressive in a conflict it entered second.
3.4 m/s² is the comfortable braking of the AASHTO Green Book and what DRAC
takes as a conflict (3.35 m/s², Archer 2005); sources flagged secondary in
the research pass are to be checked before they are cited.

## How the labels were made

1. `scripts/responsibility/label_ooi_aggression.py` computes the model-free
   evidence of each scene in both directions: right of way (rule-based
   reading), footprint PET, a_req, exact box gaps, headways, lane changes
   against the other's path, the light or stop sign each car faced over time,
   and a figure per scene with the map, both tracks, speeds, gap, headway,
   the PET band and the control bars.
2. Lane pass: the 500 figures were judged by eye by ten labellers working on
   50 scenes each, and 16 borderline rows were made uniform
   (`lane_pass_labels.csv`).
3. Right-of-way pass: six annotators recorded the facts for the 257
   conflicts (about 43 each, from the figures with the rule candidates hidden;
   `facts/main_*.csv`). The verdicts come from the script alone.
4. Two further annotators relabelled 48 conflicts independently
   (`facts/dup_*.csv`; `agreement.csv`).

The rules alone were not good enough to serve as labels. The right-of-way
reading of `right_of_way.py` named a holder in 140 of the 257 conflicts. Of
those, the annotators agreed in 128, named the other car in 8, and found no
holder in 4. In the 100 other conflicts the rule found nothing and the
annotators could (left turns across oncoming traffic, arrival order at
all-way stops, driveways). The recurring errors: a turn read as a lane
change, adjacent or oncoming lanes read as one lane, roundabouts, the first
car at an all-way stop, a left-turner read as the other car, stop signs and
lights of another approach, and tracking spikes read as braking.

## The labels

| | aggressive | cut-in | tailgate | close pass | took the right of way | signal or sign |
|---|---|---|---|---|---|---|
| SDC toward OOI | 8 | 8 | 0 | 0 | 0 | 0 |
| OOI toward SDC | 63 | 50 | 11 | 1 | 0 | 1 |

- **Both aggressive** in one scene: 482. The SDC cut in, and the OOI then held a 0.45 s headway for 4.5 s.
- **Right-of-way positive:** only OOI 128 (a right turn on red without stopping, ahead of the SDC's U-turn on green; checked by a person). Close pass: OOI 271, in the lane pass.
- **Conflicts:** of the 257, F held the right of way in 210 and lacked it in 26 (the rest: none or unclear). Of those 26, 7 were closer than 1.5 s. F broke its own control in 28 (annotators' reading; the rule keeps fewer, see above). S reacted hard (≥ 3.4 m/s²) in 4 (13, 130, 226, 426), and in none of them to a near stop.

Logged driving rarely takes the right of way. Of 257 conflicts the verdict
is aggressive in 1 (128). A first version of the rules gave 8; eight of
those were reviewed by a person (below) and seven were not aggressive, which
led to the refinements above: passes of cars turning past each other, stop
signs matched to the wrong approach, a left turn whose own light is not in
the map, and a hard reaction that did not cost the other its speed. S slowed by at least 1 m/s in 65 conflicts. In 48 of them F
held the right of way, so S was yielding as it should. The annotators
attributed S's slowing to F in 13, to S's own turn in 12, to its own red
or stop in 3, and recorded no reaction in 37 (S had slowed earlier, or was
creeping, or accelerating within their window; those numbers stay in
`conflicts.csv`).

## Checked by a person

A person went through the first version's right-of-way positives
(`human_review.csv`; `human_checked` and `human_note` in the labels): 13, 62,
163, 304, 343, 360 and 391 not aggressive, 128 aggressive. The reasons: in 13 no hard braking is visible and the SDC had cleared the
junction before the OOI turned; in 62 and 343 the OOI was turning and the SDC's
right turn is fine (the tracks never cross); in 163 the OOI waited at a
stop sign for the through traffic; in 360 the distance is too far; in 304
the OOI's light is not in the map but it was allowed to turn left. The
rules were then refined (above) and now give the same verdict on all eight.
They agree with the person by construction, since the person's corrections
shaped them: a check on fresh conflicts is still missing.

## Agreement

48 conflicts were annotated twice, independently (`agreement.csv`):

| item | agreement | Cohen's κ |
|---|---|---|
| holder (sdc / ooi / none / unclear) | 0.88 | 0.76 |
| first car broke its control | 0.88 | 0.78 |
| why the second car slowed | 0.67 | 0.19 |
| verdict (first car aggressive) | 1.00 | 1.00 |

The verdict's κ means little: the duplicate subset holds one positive,
and both annotators found it. The informative numbers are the first two:
the facts the verdict rests on are agreed on to substantial but not
near-perfect measure. The cause of a slowdown is the weakest fact (κ 0.19).
It matters only where the reaction is hard or the control broken, which were
settled by reading the figures. The agreement rests on two annotators who
are the same kind of model as the first six, so it measures consistency, not
validity. Human checks of the borderline rows listed below are still
needed.

## Debatable rows

- 128: the OOI's red-light crossing at 2.5 m/s (checked by a person: aggressive).
- Cut-ins at 0.8–0.9 s by a faster car that pulls away; left-turners that start before the other car has cleared and pass it at 0.5–0.85 m (labelled not aggressive: 246, 256, 258, 418, 428).
- 94 conflicts are flagged `needs_review` in `row_labels.csv` (low confidence, unclear holder, or a reaction the annotator did not record). None changes a verdict: only a hard reaction or a stop could, and there are 4 hard reactions and no stops (see Conflicts).
- Scenes 105/311 and 377/447/465 share a scenario ID; 342/482 and 466/474 are overlapping clips of one drive (482 sees 2 s more).

## Columns

| column | meaning |
|---|---|
| `scene`, `scenario_id`, `sdc_id`, `ooi_id` | file stem in `raw_scenes_500`; WOMD scenario; track IDs of the pair |
| `sdc_aggressive`, `ooi_aggressive` | 1 when that car was aggressive toward the other |
| `sdc_kind`, `ooi_kind` | cut_in, tailgate, close_pass, took_row, signal_or_sign or margin |
| `sdc_reason`, `ooi_reason` | one sentence with the deciding numbers and times |
| `conflict_*` | for conflicts: which car went first, PET, crossing angle, holder and basis, whether F broke its control, S's measured reaction and cause, the rule applied |

To check a label, look at the figure and video. They show the labels and
reasons in the caption, the PET as a red band, and the control each car faced:

```bash
python -m scripts.responsibility.label_ooi_aggression --scenes raw_scenes_500 --out-dir logs/responsibility/ooi_labels_v2 \
    --labels responsibility/labels/raw_scenes_500_ooi_aggression.csv --video --workers 8
# logs/responsibility/ooi_labels_v2/{evidence.csv, figures/<scene>.png, videos/<scene>.mp4}
python -m scripts.responsibility.derive_row_labels --conflicts responsibility/labels/conflicts.csv \
    --facts "responsibility/labels/facts/main_*.csv" --duplicates "responsibility/labels/facts/dup_*.csv" \
    --lane-labels responsibility/labels/lane_pass_labels.csv --out-dir OUT   # rederive from the facts
```
