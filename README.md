# Shu 2al

Shu 2al → Shoo Ahl (ʃu ʔaːl) - شو قال - is a Levantine Arabic phrase that translates literally to "What did [he] say?".

Shu 2al is a subtitle generating and translating project. You can load a YouTube playlist, and it will download it, generate subtitles, then translate them to English, and maybe to the language of your choice in the future.

All you need is a powerful GPU, some time, and patience.

## Why Rust?

Cause I hate Python (I haven't written Python in months and I'm just too lazy to get back), and because I'm just too used to the Rust ecosystem.

## How do I run it?

For now, you can only run it through CLI. I'm planning to make a front-end for this later and potentially host it online as a service.

## Why Make This?

Because I love Syrian drama and I think it should be more accessable to non-Arabic speakers.

## My Vision

Currently, the configuration is hard-coded so you would need to modify the source code and recompile to fine tune the program for your own needs.

This program is going to be made of modules (or blocks) and you will be able to fine-tune each module so that it works best with your language and dialect.

### Source

#### File

Simple `.srt` file loader.

#### Model

Even though there is only one model, this paves the way for future models to be added.

##### Whisper

`whisper` module consumes video or audio stream and generates a simple .srt file, this is the foundation block, which you can customize how the `.srt` file is handled in down-stream modules.

### Modules

#### Trimmer (Optional)

the trimmer module will take an `.srt` file and apply some filters such as:

* **Word filter**: any AI could hallucinate and generate subtitles that are commonly found in community-made subtitles, such as "Subscribe!" and "Translated by [name]"
* **trimmer**: removes empty subtitles

#### Synchronizer (Optional)

The synchronizer module will take an `.srt` file and try to match it with the audio stream, so that the subtitles appear when a person speaks.

#### Translater (Optional)

The translator module simply translates the subtitles to the language of your choice

### Target

Where to save the output `.srt` file.

## What do we Have Currently?

I'm still laying the ground work for the modules. There is some existing code for the trimmer module, but all needs to be heavily refactored to work with this new design.
