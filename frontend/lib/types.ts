// Contract with the FastAPI backend (/compare and /search).
// spans = [start, end) character offsets within the matching sentence.
export type Span = [number, number];

export type Match = {
  sentence: string;       // sentence from the submitted text
  source: string;         // source sentence that was found
  source_label?: string;  // name of the source document
  probability: number;    // 0..1, calibrated probability of copying
  spans: Span[];          // copied fragments in `sentence`
  source_spans: Span[];   // copied fragments in `source`
};

export type Result = {
  probability: number;    // overall probability of infringement
  total_sentences: number;
  matches: Match[];
  elapsed_ms: number;
};
