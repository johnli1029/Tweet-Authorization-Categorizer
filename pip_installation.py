try:
    from pip import main as pipmain
except:
    from pip._internal import main as pipmain


if __name__ == '__main__':
    pipmain(['install', "--upgrade", "pip"])
    modules = ['numpy', 'pandas', 'sklearn', 'nltk']
    for module in modules:
        pipmain(['install', module, '--user'])


