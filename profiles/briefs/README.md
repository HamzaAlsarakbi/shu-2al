# Show briefs

A brief gives the translator context it can't get from the audio: the premise, the
characters and how they're related, and the spelling of names in Arabic and English.
The translator sends it with every batch, so names stay consistent and the model can
tell who is talking to whom.

- **Server:** put the file at `<data>/briefs/<series>.md`, where `<series>` is the
  series name the queue parsed from the title (e.g. `الندم.md`). It's picked up
  automatically for every episode of that series.
- **CLI:** `shu2al run … --brief path/to/brief.md`.

Keep it short: a few sentences of premise plus a character list.
