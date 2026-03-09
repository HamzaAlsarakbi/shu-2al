use crate::{
    core::{language::Language, srt::SRT},
    source::source::Source,
};
/// Whisper model type
///
/// See https://github.com/openai/whisper/blob/main/model-card.md#model-details for details about each model.
#[derive(Debug, Clone, Default)]
pub enum WhisperModel {
    Tiny,
    Base,
    Small,
    #[default]
    Medium,
    Large,
    LargeV2,
    LargeV3,
    Turbo,
}
/// Whisper task type
#[derive(Debug, Clone, Default)]
pub enum WhisperTask {
    #[default]
    Transcribe,
    Translate,
}

/// Used to configure the Whisper model for inference.
/// It contains various parameters that control the behavior of the model.
/// The default values are set to the recommended values for the model.
pub struct WhisperConfig {
    /// Input audio file to process
    file: String,
    /// Model to use for inference
    model: WhisperModel,
    /// Language of the input audio; if not provided (default: `English`)
    language: Language,
    /// Task to perform; `transcribe` for transcription and `translate` for
    /// translation; if not provided, the model will automatically detect the
    /// language and perform the task accordingly (default: `transcribe`)
    task: WhisperTask,
    /// Temperature to use for sampling (default: `0`)
    temperature: f32,
    /// number of candidates when sampling with non-zero temperature
    /// (default: `5`)
    best_of: u32,
    /// Number of beams in beam search, only applicable when temperature is
    /// zero (default: `5`)
    beam_size: u32,
    /// Optional patience value to use in beam decoding, as in
    /// https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to
    /// conventional beam search (default: `None`)
    patience: Option<f32>,
    /// Optional token length penalty coefficient (alpha) as
    /// in https://arxiv.org/abs/1609.08144, uses simple length normalization
    /// by default (default: `None`)
    length_penalty: Option<f32>,
    /// Comma-separated list of token ids to suppress during sampling; '-1'
    /// will suppress most special characters except common punctuations
    /// (default: `-1`)
    supress_tokens: String,
    /// Optional text to provide as a prompt for the first window. (default: `None`)
    initial_prompt: Option<String>,
    /// If `True`, provide the previous output of the model as a prompt for the
    /// next window; disabling may make the text inconsistent across windows,
    /// but the model becomes less prone to getting stuck in a failure loop
    /// (default: `True`)
    condition_on_previous_text: bool,
    /// Whether to perform inference in fp16; True by default
    /// (default: `True`)
    fp16: bool,
    /// Temperature to increase when falling back when the decoding fails to
    /// meet either of the thresholds below (default: `0.2`)
    temperature_increment_on_fallback: f32,
    /// If the gzip compression ratio is higher than this value, treat the
    /// decoding as failed (default: `2.4`)
    compression_ratio_threshold: f32,
    /// If the average log probability is lower than this value, treat the
    /// decoding as failed (default: `-1.0`)
    logprob_threshold: f32,
    /// If the probability of the <|nospeech|> token is higher than this value
    /// AND the decoding has failed due to `logprob_threshold`, consider the
    /// segment as silence (default: `0.6`)
    no_speech_threshold: f32,
    /// (Experimental) Extract word-level timestamps and refine the results
    /// based on them (default: `False`)
    word_timestamps: bool,
    /// If `word_timestamps` is `True`, merge these punctuation symbols with
    /// the next word (default: "'“¿([{-)
    prepend_punctuation: String,
    /// If `word_timestamps` is `True`, merge these punctuation symbols with
    /// the previous word (default: "'.。,，!！?？:：”)]}、)
    append_punctuation: String,
    /// (Requires `word_timestamps` `True`) Underline each word as it is spoken
    /// in srt and vtt (default: `False`)
    highlight_words: bool,
    /// (Requires `word_timestamps` `True`) The maximum number of characters in
    /// a line before breaking the line (default: `None`)    
    max_line_width: Option<u32>,
    /// (Requires `word_timestamps` `True`) The maximum number of lines in a
    /// segment (default: `None`)
    max_line_count: u32,
    /// (Requires `word_timestamps` `True`, No effect with `max_line_width`)
    /// the maximum number of words in a segment (default: `None`)
    max_words_per_line: Option<u32>,
    /// Number of threads used by torch for CPU inference; supercedes
    /// MKL_NUM_THREADS/OMP_NUM_THREADS (default: `0`)
    threads: u32,
    /// Comma-separated list start,end,start,end,... timestamps (in seconds) of
    /// clips to process, where the last end timestamp defaults to the end of
    /// the file (default: `0`)
    clip_timestamps: u32,
    /// (Requires `word_timestamps` `True`) Skip silent periods longer than
    /// this threshold (in seconds) when a possible hallucination is detected
    /// (default: `None`)
    hallucination_silence_threshold: Option<f32>,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            file: String::new(),
            model: WhisperModel::default(),
            language: Language::default(),
            task: WhisperTask::default(),
            temperature: 0.0,
            best_of: 5,
            beam_size: 5,
            patience: None,
            length_penalty: None,
            supress_tokens: "-1".to_string(),
            initial_prompt: None,
            condition_on_previous_text: true,
            fp16: true,
            temperature_increment_on_fallback: 0.2,
            compression_ratio_threshold: 2.4,
            logprob_threshold: -1.0,
            no_speech_threshold: 0.6,
            word_timestamps: false,
            prepend_punctuation: "'“¿([{-".to_string(),
            append_punctuation: "'.。,，!！?？:：”)]}、".to_string(),
            highlight_words: false,
            max_line_width: None,
            max_line_count: 0,
            max_words_per_line: None,
            threads: 0,
            clip_timestamps: 0,
            hallucination_silence_threshold: None,
        }
    }
}

impl Source for WhisperConfig {
    fn process(
        input: crate::source::source::SourceInput,
    ) -> Result<SRT, crate::core::error::SRTError> {
        todo!()
    }
}
