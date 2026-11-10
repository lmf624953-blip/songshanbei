"""Decode media independently of the generating process's exit status."""
from pathlib import Path


def validate_generated_media(path, config):
    import av
    path = Path(path)
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1 or len(container.streams.audio) != 1:
            raise ValueError('Expected one video stream and one audio stream')
        stream = container.streams.video[0]
        video = dict(width=stream.width, height=stream.height,
                     fps=float(stream.average_rate), codec=stream.codec_context.name,
                     frames=sum(1 for _ in container.decode(video=0)))
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        audio = dict(codec=stream.codec_context.name, sample_rate=stream.codec_context.sample_rate,
                     channels=stream.codec_context.channels, decoded_frames=0, decoded_samples=0)
        for frame in container.decode(audio=0):
            audio['decoded_frames'] += 1
            audio['decoded_samples'] += frame.samples
    passed = (video['frames'] == config['num_frames']
              and abs(video['fps'] - config['fps']) < 0.01
              and video['codec'] == 'h264'
              and (video['height'], video['width']) == (config['height'], config['width'])
              and audio['codec'] == 'aac' and audio['sample_rate'] == 32000
              and audio['channels'] == 2 and audio['decoded_samples'] > 0)
    return dict(media_readable=True, video=video, audio=audio, video_specs_passed=passed,
                similarity_evaluated=False, competition_validation_passed=False)
