# Define answer schema to be used in final LLM call format option
# Just used for testing
whatever_answer_schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"}
    },
    "required": ["answer"],
    "additionalProperties": False
}

bool_answer_schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "answer_bool": {"type": "boolean"}
    },
    "required": ["answer", "answer_bool"],
    "additionalProperties": False
}

ranking_schema = {
  "type": "object",
  "properties": {
    "ranking": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "properties": {
          "procedure_index": {"type": "integer", "minimum": 0},
          "rank": {"type": "integer", "minimum": 1},
          "score": {"type": "number", "minimum": 0, "maximum": 10},
          "reasons": {"type": "array", "items": {"type": "string", "maxLength":1000}, "maxItems":10}
        },
        "required": ["procedure_index", "rank", "score", "reasons"]
      }
    },
  },
  "required": ["ranking"],
  "additionalProperties": False
}

# ranking_schema = {
#   "type": "object",
#   "properties": {
#     "ranking": {
#       "type": "array",
#       "minItems": 1,
#       "items": {
#         "type": "object",
#         "properties": {
#           "procedure_index": {"type": "integer", "minimum": 0},
#           "rank": {"type": "integer", "minimum": 1},
#           "score": {"type": "number", "minimum": 0, "maximum": 10},
#           "reasons": {"type": "array", "items": {"type": "string"}},
#           "flags": {"type": "array", "items": {"type": "string"}}
#         },
#         "required": ["procedure_index", "rank", "score", "reasons"]
#       }
#     },
#     "best_summary": {"type": "string"},
#     "worst_summary": {"type": "string"}
#   },
#   "required": ["ranking"]
# }


# Force answer to have string answer, but want final numerical value for comparison
GSM_answer_schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "maxLength":1000},
        "answer_numerical": {"type": "number"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["answer", "answer_numerical"],
    "additionalProperties": False
}

# Force answers to be single letter, allow for optional confidence interval if asked for
ARC_answer_schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
        "confidence": {"type": "number"}
    },
    "required": ["answer"],
    "additionalProperties": False
}