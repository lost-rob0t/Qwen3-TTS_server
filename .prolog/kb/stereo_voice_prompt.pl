voice_prompt_invariant(reference_audio, mono_float32).
upstream_failure(stereo_reference_audio,
                 tuple_item_assignment_in_qwen_audio_normalizer).
regression_test(stereo_reference_audio,
                'tests/test_server_audio.py').
