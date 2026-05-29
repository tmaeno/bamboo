"""Mattermost frontend for bamboo (ops-facing chat).

This subpackage holds the Mattermost presentation adapter and bot daemon.  It is
**optional**: core bamboo never imports it, and the third-party Mattermost client
lives behind the ``bamboo[mattermost]`` extra.  Modules that need the client
import it lazily (see :mod:`bamboo.frontends.mattermost.driver`) so importing this
package without the extra installed does not fail until a connection is actually
attempted.

Layout (mirrors the plan):

* ``render``  — pure message/attachment builders (no network, no client).
* ``driver``  — lazy client factory (raises a friendly error if the extra is missing).
* ``poster``  — Phase 1 wedge: post an :class:`AnalysisResult` to a channel.
"""
