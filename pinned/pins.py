"""Every pin of the v4 worker in one place. CANDIDATE values: see docs/V4_PINNING.md for how each was established and what is still unverified."""

# --- Chatterbox source (pip installs it from this commit with --no-deps; see Dockerfile)
CHATTERBOX_REPO = "https://github.com/resemble-ai/chatterbox.git"
CHATTERBOX_COMMIT = "5de7a54aa4e5e2baadb0182dde554908b48b85c2"
CHATTERBOX_DIST = "chatterbox-tts"
CHATTERBOX_VERSION = "0.1.7"

# --- Perth (the watermarker). Upstream declares it as `@master`, a moving ref: pinned to the commit master had on 2026-10-28.
PERTH_REPO = "https://github.com/resemble-ai/Perth.git"
PERTH_COMMIT = "ff1c8ac55a976971245cdd53c18d6131ca00d993"
PERTH_DIST = "resemble-perth"
PERTH_VERSION = "1.1.0"

# --- Runtime libraries that decide whether the pinned Chatterbox behaves as reviewed
TORCH_VERSION = "2.7.1"            # the approved Docker baseline; Chatterbox itself declares torch==2.6.0 (an open question, see the docs)
TORCHAUDIO_VERSION = "2.7.1"
TRANSFORMERS_VERSION = "5.2.0"

# --- Checkpoint: Hugging Face repo revision (never "main")
CHECKPOINT_REPO = "ResembleAI/chatterbox"
CHECKPOINT_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
T3_FILE = "t3_mtl23ls_v3.safetensors"       # the V3 T3 weights. The V2 file (t3_mtl23ls_v2.safetensors) must never be loaded.
V2_T3_FILE = "t3_mtl23ls_v2.safetensors"
S3GEN_FILE = "s3gen.pt"                      # the pinned source loads s3gen.pt, NOT s3gen_v3.*

# file -> (size in bytes, sha256). The four LFS files carry the sha256 published by the Hub for this revision; the two small JSONs are not LFS, so their sha256 was computed
# from the downloaded bytes and corroborated by their git blob ids (grapheme d27fb3f2fd38ca39b7bbfbf83a13b3c617e551df, Cangjie d77891f84ca1db0d6f7058a4ee081d4bb0bfe88e) matching the Hub API.
CHECKPOINT_FILES = {
    "t3_mtl23ls_v3.safetensors": (2143989928, "5abca8321ede76f8e61f1cc0d19aea6c946b28871017ce8726f8a69203f05953"),
    "s3gen.pt": (1057165844, "9b9ff07e60b20c136e2b1b3d7563a24604e8d2c4c267888d1ee929dd0151d2a3"),
    "ve.pt": (5698626, "4b16d836bc598509860f6fa068165a8bb5e9ac84f05582dfcf278a5a372879f1"),
    "conds.pt": (107374, "6552d70568833628ba019c6b03459e77fe71ca197d5c560cef9411bee9d87f4e"),
    "grapheme_mtl_merged_expanded_v1.json": (69989, "69632f47220a788a52ce2661d096453c5655e9bf25289d89a8d832c46ee07dbf"),
    "Cangjie5_TC.json": (1920163, "7073fd9de919443ae88e0bd2449917a65fe54898a4413ed1edcc4b67f28bce8c"),
}

# --- Tokenizer assets that Chatterbox would otherwise fetch from the network at run time (pinned/assets.py makes them local and verified)
# (1) Cangjie5_TC.json: tokenizer.py calls hf_hub_download(repo_id="ResembleAI/chatterbox", filename="Cangjie5_TC.json", cache_dir=<checkpoint dir>) with NO revision.
#     The image gives that call a Hugging Face cache laid out inside the checkpoint dir, with refs/main -> CHECKPOINT_REVISION, so it resolves the pinned file offline.
CANGJIE_FILE = "Cangjie5_TC.json"                    # its size and sha256 are in CHECKPOINT_FILES
CANGJIE_DISTINCT_WORDS = 99228                       # distinct words in the pinned file (126610 entries): what ChineseCangjieConverter.word2cj must hold once loaded
# (2) spacy_pkuseg.pkuseg() (default model "spacy_ontonotes") downloads a zip from GitHub when <PKUSEG_HOME>/spacy_ontonotes.zip is absent (it checks that file only, not the extracted folder).
#     The sha256 is the one shipped inside the pinned spacy_pkuseg==1.0.2 (config.model_hash["spacy_ontonotes"]); it equals an independent sha256sum of the downloaded file.
#     The GitHub release (v0.0.26, tag commit 9f14dc133897cc30a5770e8892f58aaf27cb8db1, published 2020-10-05) is NOT marked immutable and has no asset digest: the hash is the control.
PKUSEG_MODEL = "spacy_ontonotes"
PKUSEG_URL = "https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip"
PKUSEG_ARCHIVE = (34567143, "b216e7f92de7ae285aeab8feba2faa8ea8216e5995ff6fb3d391cc8356db1bfe")     # (size, sha256) of spacy_ontonotes.zip
PKUSEG_FILES = {                                     # the ONLY entries of the archive: path inside <PKUSEG_HOME>/spacy_ontonotes/ -> (size, sha256), hashed from the verified archive
    "features.msgpack": (22685181, "fd4322482a7018b9bce9216173ae9d2848efe6d310b468bbb4383fb55c874a18"),
    "weights.npz": (37508754, "5ada075eb25a854f71d6e6fa4e7d55e7be0ae049255b1f8f19d05c13b1b68c9e"),
}
PKUSEG_DIST_VERSION = "1.0.2"                        # spacy_pkuseg: its model_hash / URL table is the source of the pin above
DEFAULT_PKUSEG_HOME = "/app/pkuseg"

# --- Canonical fingerprints (pinned/fingerprint.py) of the upstream functions the worker's behaviour depends on, taken from the source at CHATTERBOX_COMMIT.
# value: (module file relative to the chatterbox package, fingerprint)
SOURCE_FINGERPRINTS = {
    "ChatterboxMultilingualTTS.from_local": ("mtl_tts.py", "9807208975fcaca25ab932affc3d2922e5979fd9175f0476a232abeba88b99b4"),
    "ChatterboxMultilingualTTS.from_pretrained": ("mtl_tts.py", "dae328f6f47de750038384b8bc14f71fd666c1c9c763fc1a233eef48b363ca70"),
    "ChatterboxMultilingualTTS.generate": ("mtl_tts.py", "48d349eb4c0149653690703a8095b76aae00b40d936dfec2e7c272dbf78dcc40"),
    "ChatterboxMultilingualTTS.__init__": ("mtl_tts.py", "08dbaf1ca0144e06c8ca7b27b07dc6a8b7afcc1bee89772dd9f1519c336e3bc5"),
    "_resolve_multilingual_t3_model": ("mtl_tts.py", "19848aed19e7598e7eb7d182aa32a38bb74f456df3097a7f9d034dd143a82884"),
    "T3.inference": ("models/t3/t3.py", "f56e58ccc9341f835598719724bd6088e5e54192b0171c43f10ff8c687bd9a07"),
    "CausalMaskedDiffWithXvec.inference": ("models/s3gen/flow.py", "43bf7163a60243ce65c53a931530c57d119fb5b200106b5df5aa81a07ba6bafe"),
    "S3Token2Mel.forward": ("models/s3gen/s3gen.py", "494c86e018f02e5fcff5828252e4bc83ef7a5268c1edf4b9575ecdc13201ad1e"),
    "S3Token2Wav.flow_inference": ("models/s3gen/s3gen.py", "5f5f7092a29783ebf55087cc662037b49396617d917a67e2bd3f2feb657d6fd3"),
    "S3Token2Wav.hift_inference": ("models/s3gen/s3gen.py", "4289fbdcd3a999e8f5a9cd58830b511aafde2156ae5a8cd6dcc11b1277c3e83e"),
    "S3Token2Wav.inference": ("models/s3gen/s3gen.py", "66bf741a3c676d0c04d5786994bc1a8c3b0fecb79ceda3d0841cc6ccfd620958"),
}

# Streamed audio would leave the worker WITHOUT the Perth watermark (upstream watermarks only the whole finished waveform, and no chunked method is verified).
# A constant on purpose: no environment variable or request field can turn it on. Flipping it is a reviewed source change made after a verified method exists.
STREAMING_ALLOWED = False
