from sglang.private.inc_tokenizer.cache import TokenizerWrapper
from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager as TokenizerManagerSRT,
)
from sglang.srt.server_args import PortArgs, ServerArgs


class TokenizerManager(TokenizerManagerSRT):

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        super().__init__(server_args, port_args)
        if server_args.enable_inc_tokenizer:
            wrapper = TokenizerWrapper(
                self.tokenizer,
                verify_tokenization_correctness=server_args.verify_inc_tokenization_correctness,
            )
            # Store original method
            self.tokenizer._original_encode = self.tokenizer.encode
            self.tokenizer._original_apply_chat_template = (
                self.tokenizer.apply_chat_template
            )

            # Replace encode with cached version
            self.tokenizer.encode = wrapper.encode
            self.tokenizer.apply_chat_template = wrapper.apply_chat_template
