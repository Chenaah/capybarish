"""
Service registry for robot middleware.

This module provides service discovery, registration, and health monitoring
for robot modules and components.

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


class ServiceStatus(Enum):
    """Service status enumeration."""

    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"


class ServiceType(Enum):
    """Service type enumeration."""

    ROBOT_MODULE = "robot_module"
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    COMMUNICATION = "communication"
    DATA_SOURCE = "data_source"
    DASHBOARD = "dashboard"
    LOGGER = "logger"


@dataclass
class ServiceInfo:
    """Information about a registered service."""

    service_id: str
    name: str
    service_type: ServiceType
    status: ServiceStatus = ServiceStatus.STARTING
    metadata: Dict[str, Any] = field(default_factory=dict)
    endpoints: Dict[str, str] = field(default_factory=dict)
    health_check_url: Optional[str] = None
    last_heartbeat: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "service_id": self.service_id,
            "name": self.name,
            "service_type": self.service_type.value,
            "status": self.status.value,
            "metadata": self.metadata,
            "endpoints": self.endpoints,
            "health_check_url": self.health_check_url,
            "last_heartbeat": self.last_heartbeat,
            "created_at": self.created_at,
            "tags": list(self.tags),
        }


class HealthChecker(ABC):
    """Abstract base class for service health checkers."""

    @abstractmethod
    async def check_health(self, service: ServiceInfo) -> ServiceStatus:
        """Check the health of a service."""
        pass


class RobotModuleHealthChecker(HealthChecker):
    """Health checker for robot modules."""

    def __init__(self, communication_manager):
        self.comm_manager = communication_manager

    async def check_health(self, service: ServiceInfo) -> ServiceStatus:
        """Check health of a robot module."""
        if "module_id" not in service.metadata:
            return ServiceStatus.UNHEALTHY

        module_id = service.metadata["module_id"]
        module_status = self.comm_manager.get_module_status(module_id)

        if module_status is None:
            return ServiceStatus.STOPPED
        elif module_status.value == "connected":
            return ServiceStatus.HEALTHY
        elif module_status.value == "pending":
            return ServiceStatus.DEGRADED
        else:
            return ServiceStatus.UNHEALTHY


class ServiceRegistry:
    """
    Service registry for managing robot services and components.

    Provides service discovery, registration, health monitoring,
    and event notifications for a distributed robot system.
    """

    def __init__(self, heartbeat_timeout: float = 30.0, health_check_interval: float = 10.0):
        """
        Initialize service registry.

        Args:
            heartbeat_timeout: Time in seconds before a service is considered stale
            health_check_interval: Interval in seconds between health checks
        """
        self.heartbeat_timeout = heartbeat_timeout
        self.health_check_interval = health_check_interval

        # Service storage
        self.services: Dict[str, ServiceInfo] = {}
        self.services_by_type: Dict[ServiceType, Set[str]] = {}
        self.services_by_tag: Dict[str, Set[str]] = {}

        # Health checking
        self.health_checkers: Dict[ServiceType, HealthChecker] = {}

        # Event callbacks
        self.on_service_registered: List[Callable[[ServiceInfo], None]] = []
        self.on_service_deregistered: List[Callable[[ServiceInfo], None]] = []
        self.on_service_status_changed: List[
            Callable[[ServiceInfo, ServiceStatus, ServiceStatus], None]
        ] = []

        # Background threads
        self._health_check_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            "total_registrations": 0,
            "total_deregistrations": 0,
            "health_checks_performed": 0,
            "services_cleaned_up": 0,
        }

    def register_service(
        self,
        name: str,
        service_type: ServiceType,
        metadata: Optional[Dict[str, Any]] = None,
        endpoints: Optional[Dict[str, str]] = None,
        tags: Optional[Set[str]] = None,
        health_check_url: Optional[str] = None,
    ) -> str:
        """
        Register a new service.

        Args:
            name: Service name
            service_type: Type of service
            metadata: Additional service metadata
            endpoints: Service endpoints
            tags: Service tags for categorization
            health_check_url: URL for health checking

        Returns:
            Unique service ID
        """
        with self._lock:
            service_id = str(uuid.uuid4())

            service = ServiceInfo(
                service_id=service_id,
                name=name,
                service_type=service_type,
                metadata=metadata or {},
                endpoints=endpoints or {},
                tags=tags or set(),
                health_check_url=health_check_url,
            )

            # Store service
            self.services[service_id] = service

            # Update indices
            if service_type not in self.services_by_type:
                self.services_by_type[service_type] = set()
            self.services_by_type[service_type].add(service_id)

            for tag in service.tags:
                if tag not in self.services_by_tag:
                    self.services_by_tag[tag] = set()
                self.services_by_tag[tag].add(service_id)

            # Update statistics
            self.stats["total_registrations"] += 1

            # Notify callbacks
            for callback in self.on_service_registered:
                try:
                    callback(service)
                except Exception as e:
                    print(f"Error in service registered callback: {e}")

            print(
                f"[ServiceRegistry] Registered service: {name} ({service_type.value}) with ID {service_id}"
            )
            return service_id

    def deregister_service(self, service_id: str) -> bool:
        """
        Deregister a service.

        Args:
            service_id: Service ID to deregister

        Returns:
            True if service was deregistered, False if not found
        """
        with self._lock:
            if service_id not in self.services:
                return False

            service = self.services[service_id]

            # Remove from indices
            if service.service_type in self.services_by_type:
                self.services_by_type[service.service_type].discard(service_id)

            for tag in service.tags:
                if tag in self.services_by_tag:
                    self.services_by_tag[tag].discard(service_id)

            # Remove service
            del self.services[service_id]

            # Update statistics
            self.stats["total_deregistrations"] += 1

            # Notify callbacks
            for callback in self.on_service_deregistered:
                try:
                    callback(service)
                except Exception as e:
                    print(f"Error in service deregistered callback: {e}")

            print(f"[ServiceRegistry] Deregistered service: {service.name} with ID {service_id}")
            return True

    def heartbeat(self, service_id: str, status: Optional[ServiceStatus] = None) -> bool:
        """
        Send heartbeat for a service.

        Args:
            service_id: Service ID
            status: Optional status update

        Returns:
            True if heartbeat was recorded, False if service not found
        """
        with self._lock:
            if service_id not in self.services:
                return False

            service = self.services[service_id]
            old_status = service.status

            # Update heartbeat time
            service.last_heartbeat = time.time()

            # Update status if provided
            if status is not None and status != old_status:
                service.status = status

                # Notify status change callbacks
                for callback in self.on_service_status_changed:
                    try:
                        callback(service, old_status, status)
                    except Exception as e:
                        print(f"Error in service status changed callback: {e}")

            return True

    def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        """Get service information by ID."""
        return self.services.get(service_id)

    def find_services(
        self,
        service_type: Optional[ServiceType] = None,
        tags: Optional[Set[str]] = None,
        status: Optional[ServiceStatus] = None,
        name_pattern: Optional[str] = None,
    ) -> List[ServiceInfo]:
        """
        Find services matching criteria.

        Args:
            service_type: Filter by service type
            tags: Filter by tags (service must have all specified tags)
            status: Filter by status
            name_pattern: Filter by name pattern (simple substring match)

        Returns:
            List of matching services
        """
        with self._lock:
            candidates = set(self.services.keys())

            # Filter by type
            if service_type is not None:
                type_services = self.services_by_type.get(service_type, set())
                candidates &= type_services

            # Filter by tags
            if tags:
                for tag in tags:
                    tag_services = self.services_by_tag.get(tag, set())
                    candidates &= tag_services

            # Apply additional filters
            results = []
            for service_id in candidates:
                service = self.services[service_id]

                # Filter by status
                if status is not None and service.status != status:
                    continue

                # Filter by name pattern
                if name_pattern is not None and name_pattern not in service.name:
                    continue

                results.append(service)

            return results

    def get_healthy_services(self, service_type: Optional[ServiceType] = None) -> List[ServiceInfo]:
        """Get all healthy services, optionally filtered by type."""
        return self.find_services(service_type=service_type, status=ServiceStatus.HEALTHY)

    def add_health_checker(self, service_type: ServiceType, checker: HealthChecker) -> None:
        """Add a health checker for a service type."""
        self.health_checkers[service_type] = checker

    def start_background_tasks(self) -> None:
        """Start background health checking and cleanup tasks."""
        if self._health_check_thread and self._health_check_thread.is_alive():
            return

        self._stop_event.clear()

        # Start health check thread
        self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread.start()

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

        print("[ServiceRegistry] Started background tasks")

    def stop_background_tasks(self) -> None:
        """Stop background tasks."""
        self._stop_event.set()

        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)

        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)

        print("[ServiceRegistry] Stopped background tasks")

    def _health_check_loop(self) -> None:
        """Background health checking loop."""
        while not self._stop_event.is_set():
            try:
                self._perform_health_checks()
            except Exception as e:
                print(f"Error in health check loop: {e}")

            self._stop_event.wait(self.health_check_interval)

    def _cleanup_loop(self) -> None:
        """Background cleanup loop for stale services."""
        while not self._stop_event.is_set():
            try:
                self._cleanup_stale_services()
            except Exception as e:
                print(f"Error in cleanup loop: {e}")

            self._stop_event.wait(self.heartbeat_timeout / 2)

    def _perform_health_checks(self) -> None:
        """Perform health checks on all services."""
        with self._lock:
            for service in list(self.services.values()):
                if service.service_type in self.health_checkers:
                    try:
                        checker = self.health_checkers[service.service_type]
                        # Note: This is a simplified sync version
                        # In a real implementation, you'd want async health checks
                        old_status = service.status
                        # For now, we'll just update based on heartbeat timing
                        current_time = time.time()
                        if current_time - service.last_heartbeat > self.heartbeat_timeout:
                            new_status = ServiceStatus.UNHEALTHY
                        else:
                            new_status = ServiceStatus.HEALTHY

                        if new_status != old_status:
                            service.status = new_status
                            for callback in self.on_service_status_changed:
                                try:
                                    callback(service, old_status, new_status)
                                except Exception as e:
                                    print(f"Error in status change callback: {e}")

                        self.stats["health_checks_performed"] += 1

                    except Exception as e:
                        print(f"Error checking health for service {service.name}: {e}")

    def _cleanup_stale_services(self) -> None:
        """Clean up services that haven't sent heartbeats."""
        current_time = time.time()
        stale_services = []

        with self._lock:
            for service_id, service in self.services.items():
                if current_time - service.last_heartbeat > self.heartbeat_timeout * 2:
                    stale_services.append(service_id)

        for service_id in stale_services:
            if self.deregister_service(service_id):
                self.stats["services_cleaned_up"] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            return {
                **self.stats,
                "total_services": len(self.services),
                "services_by_type": {
                    stype.value: len(services) for stype, services in self.services_by_type.items()
                },
                "healthy_services": len(self.find_services(status=ServiceStatus.HEALTHY)),
                "unhealthy_services": len(self.find_services(status=ServiceStatus.UNHEALTHY)),
            }

    def export_services(self) -> Dict[str, Any]:
        """Export all service information."""
        with self._lock:
            return {
                "services": [service.to_dict() for service in self.services.values()],
                "statistics": self.get_statistics(),
                "exported_at": time.time(),
            }
