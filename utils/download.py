from anylearn.interfaces.resource.resource_downloader import SyncResourceDownloader
from anylearn.config import init_sdk
init_sdk('http://anylearn.nelbds.cn/', 'ZDandsomSP', 'ZDandsomSP')

downloader = SyncResourceDownloader()
downloader.run(resource_id="DSETffe28cab4802843e92a78360a580", save_path="/data/ts2b")