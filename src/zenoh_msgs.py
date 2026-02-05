"""
Zenoh message types and utilities for OM1 integration.
Uses pycdr2 for CDR serialization compatible with OM1.
"""

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum

import zenoh
from pycdr2 import IdlStruct
from pycdr2.types import int8, int32, uint32

logging.basicConfig(level=logging.INFO)


@dataclass
class Time(IdlStruct, typename="Time"):
    """Time message."""

    sec: int32
    nanosec: uint32


@dataclass
class Duration(IdlStruct, typename="Duration"):
    """Duration message."""

    sec: int32
    nanosec: uint32


@dataclass
class Header(IdlStruct, typename="Header"):
    """Standard metadata for higher-level stamped data types."""

    stamp: Time
    frame_id: str


@dataclass
class String(IdlStruct, typename="String"):
    """String message."""

    data: str


@dataclass
class PersonGreetingStatus(IdlStruct, typename="PersonGreetingStatus"):
    """Person greeting status message."""

    class STATUS(Enum):
        """
        Code enum for PersonGreetingStatus.

        APPROACHING: A person is approaching.
        APPROACHED: A person has approached.
        SWITCH: Switch state from the conversation to find the next person.
        """

        APPROACHING = 0
        APPROACHED = 1
        SWITCH = 2

    header: Header
    request_id: String
    status: int8


def create_zenoh_config(network_discovery: bool = True) -> zenoh.Config:
    """
    Create a Zenoh configuration for a client connecting to a local server.

    Parameters
    ----------
    network_discovery : bool, optional
        Whether to enable network discovery (default is True).

    Returns
    -------
    zenoh.Config
        The Zenoh configuration object.
    """
    config = zenoh.Config()
    if not network_discovery:
        config.insert_json5("mode", '"client"')
        config.insert_json5("connect/endpoints", '["tcp/127.0.0.1:7447"]')

    return config


def open_zenoh_session() -> zenoh.Session:
    """
    Open a Zenoh session with a local connection first, then fall back to network discovery.

    Returns
    -------
    zenoh.Session
        The opened Zenoh session.

    Raises
    ------
    Exception
        If unable to open a Zenoh session.
    """
    local_config = create_zenoh_config(network_discovery=False)
    try:
        session = zenoh.open(local_config)
        logging.info("Zenoh client opened without network discovery")
        return session
    except Exception:
        logging.info("Falling back to network discovery...")

    config = create_zenoh_config()
    try:
        session = zenoh.open(config)
        logging.info("Zenoh client opened with network discovery")
        return session
    except Exception as e:
        logging.error(f"Error opening Zenoh client: {e}")
        raise Exception("Failed to open Zenoh session") from e


def prepare_header(frame_id: str = "") -> Header:
    """
    Prepare a Header with the current timestamp and a given frame ID.

    Parameters
    ----------
    frame_id : str
        The frame ID to be set in the header.

    Returns
    -------
    Header
        A Header object with the current timestamp.
    """
    ts = time.time()
    remainder, seconds = math.modf(ts)
    timestamp = Time(sec=int32(seconds), nanosec=uint32(remainder * 1000000000))
    header = Header(stamp=timestamp, frame_id=frame_id)
    return header
