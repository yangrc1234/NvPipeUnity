import ffmpeg

stream = ffmpeg.input('binrayvideo', format='h264')
stream = stream.output(stream, './out.mp4')
ffmpeg.run(stream)
