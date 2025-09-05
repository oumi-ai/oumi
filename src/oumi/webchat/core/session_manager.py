# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Session management for WebChat server with concurrency control."""

import asyncio
import time
from typing import Dict, Optional, Any, Callable, Tuple, TypeVar, List, Deque, Set, Union
from collections import defaultdict, deque

from oumi.core.configs import InferenceConfig
from oumi.webchat.core.session import WebChatSession
from oumi.utils.logging import logger


T = TypeVar('T')  # Generic type for session operation results
SessionOperation = Callable[[WebChatSession], T]  # Type for session operations
AsyncSessionOperation = Callable[[WebChatSession], T]  # Type for async session operations


class SessionManager:
    """Manages WebChat session lifecycle with concurrency control."""
    
    def __init__(self, default_config: InferenceConfig, system_prompt: Optional[str] = None):
        """Initialize the session manager.
        
        Args:
            default_config: Default inference configuration for new sessions.
            system_prompt: Optional system prompt for all sessions.
        """
        self.default_config = default_config
        self.system_prompt = system_prompt
        self.sessions: Dict[str, WebChatSession] = {}
        
        # Session cleanup configuration
        self.session_cleanup_interval = 3600  # 1 hour
        self.max_idle_time = 1800  # 30 minutes
        
        # Session concurrency control
        self.locking_enabled = getattr(default_config, 'session_locking_enabled', True)
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()  # Guards lock creation
        
        # Debug logging throttling
        self._last_debug_log_time: Dict[str, float] = {}
        
        # Session metrics and monitoring
        self._lock_metrics = defaultdict(lambda: {
            'wait_times': deque(maxlen=100),  # Last 100 wait times
            'hold_times': deque(maxlen=100),  # Last 100 hold times
            'concurrent_requests': 0,         # Currently active requests
            'total_requests': 0,              # Total requests since startup
            'contention_count': 0,            # Number of times lock was contended
        })
    
    async def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Thread-safe lock creation for session.
        
        Args:
            session_id: Session identifier to get a lock for.
            
        Returns:
            An asyncio.Lock for the session.
        """
        async with self._locks_lock:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = asyncio.Lock()
            return self._session_locks[session_id]
    
    async def get_or_create_session(self, session_id: str, db=None) -> WebChatSession:
        """Get existing session or create a new one.
        
        Args:
            session_id: Unique session identifier.
            db: Optional WebchatDB instance for hydration.
        
        Returns:
            WebChatSession instance.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = WebChatSession(session_id, self.default_config)
            logger.debug(f"🆕 Created new webchat session: {session_id}")
            
            # Attempt to hydrate from database if available
            if db:
                try:
                    db_data = db.hydrate_session(session_id)
                    if db_data:
                        self.sessions[session_id].hydrate_from_db(db_data)
                    else:
                        logger.debug(f"🗄️ No persistence data found for session {session_id}")
                except Exception as hydration_error:
                    logger.warning(f"⚠️ Failed to hydrate session {session_id}: {hydration_error}")
        else:
            logger.debug(f"🔄 Using existing session: {session_id}")
        
        session = self.sessions[session_id]
        
        # Add system prompt if it's set
        if self.system_prompt is not None and not hasattr(session, 'system_prompt'):
            session.system_prompt = self.system_prompt
        
        # CRITICAL FIX: Add session integrity validation
        if not hasattr(session, 'branch_manager') or session.branch_manager is None:
            from oumi.core.commands.conversation_branches import ConversationBranchManager
            logger.error(f"🚨 CRITICAL: Session {session_id} has corrupted branch_manager! Recreating...")
            session.branch_manager = ConversationBranchManager(session.conversation_history)
            # Re-sync main branch
            if "main" in session.branch_manager.branches:
                session.branch_manager.branches["main"].conversation_history = session.conversation_history
        
        session.update_activity()
        return session
    
    async def get_or_create_session_safe(self, session_id: str, db=None) -> WebChatSession:
        """Thread-safe session creation with minimal lock time.
        
        Args:
            session_id: Unique session identifier.
            db: Optional WebchatDB instance for hydration.
            
        Returns:
            WebChatSession instance.
        """
        if not self.locking_enabled:
            return await self.get_or_create_session(session_id, db)  # Rollback path
        
        # Track concurrent access with proper cleanup
        metrics = self._lock_metrics[session_id]
        metrics['concurrent_requests'] += 1
        metrics['total_requests'] += 1
        
        try:
            if metrics['concurrent_requests'] > 1:
                metrics['contention_count'] += 1
                logger.debug(f"🔄 Concurrent session access: {session_id} ({metrics['concurrent_requests']} requests)")
            
            # Time lock acquisition
            wait_start = time.monotonic()
            session_lock = await self._get_session_lock(session_id)
            
            async with session_lock:
                wait_time = time.monotonic() - wait_start
                hold_start = time.monotonic()
                
                metrics['wait_times'].append(wait_time)
                if wait_time > 0.1:  # Log slow lock acquisition
                    logger.warning(f"⏱️  Slow lock acquisition for {session_id}: {wait_time:.3f}s")
                
                try:
                    # CRITICAL SECTION - All session operations under lock
                    if session_id not in self.sessions:
                        # Create session (fast, in-memory only)
                        self.sessions[session_id] = WebChatSession(session_id, self.default_config)
                        
                        # Add system prompt if it's set
                        if self.system_prompt is not None:
                            self.sessions[session_id].system_prompt = self.system_prompt
                            
                        logger.debug(f"🆕 Created new webchat session: {session_id}")
                        
                        # Attempt to hydrate from database if available
                        if db:
                            try:
                                db_data = db.hydrate_session(session_id)
                                if db_data:
                                    self.sessions[session_id].hydrate_from_db(db_data)
                                else:
                                    logger.debug(f"🗄️ No persistence data found for session {session_id}")
                            except Exception as hydration_error:
                                logger.warning(f"⚠️ Failed to hydrate session {session_id}: {hydration_error}")
                    
                    session = self.sessions[session_id]
                    
                    # Integrity check and fix INSIDE lock to prevent races
                    if not hasattr(session, 'branch_manager') or session.branch_manager is None:
                        from oumi.core.commands.conversation_branches import ConversationBranchManager
                        logger.error(f"🚨 Session {session_id} corrupted! Recreating branch_manager...")
                        session.branch_manager = ConversationBranchManager(session.conversation_history)
                        if hasattr(session.branch_manager, 'branches') and "main" in session.branch_manager.branches:
                            session.branch_manager.branches["main"].conversation_history = session.conversation_history
                    
                    # Update activity timestamp with monotonic time
                    session.last_activity = time.monotonic()
                    # END CRITICAL SECTION
                finally:
                    hold_time = time.monotonic() - hold_start
                    metrics['hold_times'].append(hold_time)
                    
                    if hold_time > 0.05:  # Log long critical sections
                        logger.warning(f"⏱️  Long critical section for {session_id}: {hold_time:.3f}s")
            
            return session
            
        finally:
            # Always decrement counter, even on exceptions
            metrics['concurrent_requests'] -= 1
    
    async def execute_session_operation(self, session_id: str, operation, *args, **kwargs):
        """Execute an operation atomically within a session's lock.
        
        This ensures operations like branch creation/switching, model swaps, 
        and conversation updates are atomic at the per-session level.
        
        Args:
            session_id: The session to operate on
            operation: Async callable to execute under lock
            *args, **kwargs: Arguments to pass to operation
            
        Returns:
            Result of operation
        """
        if not self.locking_enabled:
            # If locking disabled, just get session and execute
            session = await self.get_or_create_session(session_id)
            return await operation(session, *args, **kwargs)
        
        # Execute under session lock for atomicity
        session_lock = await self._get_session_lock(session_id)
        
        # Track metrics
        metrics = self._lock_metrics[session_id]
        metrics['total_requests'] += 1
        
        # Time lock acquisition
        wait_start = time.monotonic()
        
        async with session_lock:
            wait_time = time.monotonic() - wait_start
            metrics['wait_times'].append(wait_time)
            hold_start = time.monotonic()
            
            try:
                # Get session under lock
                if session_id not in self.sessions:
                    self.sessions[session_id] = WebChatSession(session_id, self.default_config)
                    
                    # Add system prompt if it's set
                    if self.system_prompt is not None:
                        self.sessions[session_id].system_prompt = self.system_prompt
                        
                    logger.debug(f"🆕 Created session {session_id} for operation")
                
                session = self.sessions[session_id]
                session.last_activity = time.monotonic()
                
                # Execute operation with session
                return await operation(session, *args, **kwargs)
            finally:
                hold_time = time.monotonic() - hold_start
                metrics['hold_times'].append(hold_time)
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions and their associated locks using safe lifecycle management."""
        if not self.sessions:
            return
        
        current_time = time.monotonic()  # Use monotonic time to avoid clock jumps
        expired_sessions = []
        
        # Phase 1: Collect potentially expired sessions (no locks held)
        for session_id, session in list(self.sessions.items()):
            if current_time - session.last_activity > self.max_idle_time:
                expired_sessions.append(session_id)
        
        if not expired_sessions:
            return
        
        logger.info(f"🧹 Cleaning up {len(expired_sessions)} expired sessions")
        
        # Phase 2: Lock and re-check each session individually
        for session_id in expired_sessions:
            try:
                # Get lock for this specific session
                session_lock = await self._get_session_lock(session_id)
                
                async with session_lock:
                    # Re-check expiration under lock (session might have been accessed)
                    if session_id not in self.sessions:
                        continue  # Already cleaned up
                    
                    session = self.sessions[session_id]
                    if current_time - session.last_activity <= self.max_idle_time:
                        continue  # Session was accessed recently, keep it
                    
                    # Session is truly expired - remove it
                    del self.sessions[session_id]
                    logger.debug(f"🗑️  Removed expired session: {session_id}")
                
                # Phase 3: Remove lock AFTER session removal (outside critical section)
                async with self._locks_lock:
                    if session_id in self._session_locks:
                        del self._session_locks[session_id]
                    if session_id in self._lock_metrics:
                        del self._lock_metrics[session_id]
                        
            except Exception as e:
                logger.error(f"🚨 Error cleaning up session {session_id}: {e}")
    
    async def start_cleanup_task(self):
        """Schedule the session cleanup task to run periodically."""
        while True:
            try:
                await asyncio.sleep(self.session_cleanup_interval)
                await self.cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
    
    def get_session_metrics(self) -> Dict[str, Any]:
        """Get session and lock metrics for monitoring."""
        active_sessions = len(self.sessions)
        active_locks = len(self._session_locks)
        
        # Calculate aggregate metrics
        total_wait_times = []
        total_hold_times = []
        concurrent_counts = []
        total_requests = 0
        contention_count = 0
        
        for session_id, metrics in self._lock_metrics.items():
            total_wait_times.extend(metrics['wait_times'])
            total_hold_times.extend(metrics['hold_times'])
            concurrent_counts.append(metrics['concurrent_requests'])
            total_requests += metrics['total_requests']
            contention_count += metrics['contention_count']
        
        # Calculate averages
        avg_wait_time = sum(total_wait_times) / max(len(total_wait_times), 1)
        avg_hold_time = sum(total_hold_times) / max(len(total_hold_times), 1)
        max_wait_time = max(total_wait_times) if total_wait_times else 0
        max_hold_time = max(total_hold_times) if total_hold_times else 0
        
        return {
            "active_sessions": active_sessions,
            "active_locks": active_locks,
            "concurrent_requests": sum(concurrent_counts),
            "total_requests": total_requests,
            "contention_count": contention_count,
            "avg_wait_time": avg_wait_time,
            "avg_hold_time": avg_hold_time,
            "max_wait_time": max_wait_time,
            "max_hold_time": max_hold_time,
            "locking_enabled": self.locking_enabled,
        }
    
    def update_context_usage(self, session_id: str):
        """Update SystemMonitor with current conversation context usage.
        
        Args:
            session_id: Session ID to update.
        """
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        try:
            # Use the existing ContextWindowManager for proper token estimation
            context_manager = session.command_context.context_window_manager
            total_tokens = 0
            
            # Check if we should log debug info (throttle to once per 60 seconds per session)
            current_time = time.time()
            should_log_debug = (
                session_id not in self._last_debug_log_time or 
                current_time - self._last_debug_log_time[session_id] >= 60.0
            )
            
            if should_log_debug:
                self._last_debug_log_time[session_id] = current_time
                logger.info(
                    f"🔍 DEBUG: Updating context usage for session {session.session_id}"
                )
                logger.info(
                    f"🔍 DEBUG: Conversation history length: {len(session.conversation_history)}"
                )
            
            for i, msg in enumerate(session.conversation_history):
                content = msg.get("content", "")
                if content:
                    msg_tokens = context_manager.estimate_tokens(content)
                    total_tokens += msg_tokens
                    if should_log_debug:
                        logger.info(
                            f"🔍 DEBUG: Message {i}: {msg_tokens} tokens, content preview: {content[:50]}..."
                        )
            
            if should_log_debug:
                logger.debug(f"🔍 DEBUG: Total tokens calculated: {total_tokens}")
                logger.debug(
                    f"🔍 DEBUG: Max context tokens: {session.system_monitor.max_context_tokens}"
                )
            
            session.system_monitor.update_context_usage(total_tokens)
            session.system_monitor.update_conversation_turns(
                len(session.conversation_history) // 2
            )
            
            # Verify the update worked (only log if debug logging is enabled for this session)
            if should_log_debug:
                stats = session.system_monitor.get_stats()
                logger.info(
                    f"🔍 DEBUG: SystemMonitor stats after update - context_used: {stats.context_used_tokens}, context_max: {stats.context_max_tokens}, percent: {stats.context_percent}"
                )
        
        except Exception as e:
            logger.warning(f"Failed to update context usage: {e}")
            import traceback
            
            logger.warning(f"Full traceback: {traceback.format_exc()}")