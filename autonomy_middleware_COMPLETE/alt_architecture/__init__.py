"""
alt_architecture — the token-emitting control model from SCR v1.0-V6.

A SECOND architecture, isolated from autonomy_middleware on purpose. In
this design the model's output is a privileged control channel; in the main
package it is data to be evaluated. Both are faithful to documents in this
project, they are not compatible, and choosing between them is a decision
about trust rather than features.

Read README.md before running anything here. Execution is disabled by
default and PURGE_THREAD has no default handler at all.
"""
from .command_listener import CommandListener, Detection, COMMANDS, SHADOW_COMMAND_NOTE
from .behavioral_protocols import (HonestRejection, BaselineHold, PriorityAlpha,
                                    RejectionState, REJECTION_OUTPUT,
                                    ACCEPTANCE_OUTPUT, PRIORITY_ALPHA_OUTPUT)
