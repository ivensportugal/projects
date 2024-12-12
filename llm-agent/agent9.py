from agent8 import HFModelDownloadsTool

tool = HFModelDownloadsTool()
print(tool.forward('text-classification'))

tool.push_to_hub('ivensportugal/hf-model-downloads', token='')