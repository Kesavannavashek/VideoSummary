import yt_dlp


def extract_video_info(video_url):

    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'format': 'bestvideo+bestaudio/best',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
    return info


def get_video_direct_url(info):

    if 'url' in info:
        return info['url']


    formats = info.get('formats', [])
    for f in formats:
        if f.get('vcodec') != 'none' and f.get('acodec') != 'none' and f.get('protocol') == 'https':
            return f['url']
    return None


