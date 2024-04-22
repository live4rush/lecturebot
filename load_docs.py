from langchain_community.document_loaders import TextLoader, DirectoryLoader


def load_docs(pickle=False):
    if pickle:
        loader = DirectoryLoader(
            "./ClassTranscriptions",
            glob="**/*.pkl",
            recursive=True,
            loader_cls=TextLoader
        )
    else:
        # Load docs
        loader = DirectoryLoader(
            "./ClassTranscriptions",
            glob="**/*.txt",
            recursive=True,

            loader_cls=TextLoader
        )

    docs = loader.load()


    return docs
