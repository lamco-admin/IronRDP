# ironrdp-rdpeudp

RDPEUDP2 wire protocol implementation for [IronRDP](https://github.com/Devolutions/IronRDP).

Implements the reliable UDP transport defined in [MS-RDPEUDP](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpeudp/), targeting version 2 (RDPEUDP2) of the protocol.

## Protocol Coverage

- SYN/SYN+ACK/ACK three-way handshake (v1 format for initial exchange)
- Version negotiation via SYNDATAEX (advertise v2, transition to RDPEUDP2 wire format)
- RDPEUDP2 packet framing (PacketPrefixByte + type-specific headers)
- Sequence numbering and acknowledgment
- Retransmission with timer-based loss recovery (min 300ms for v2)
- NewReno-variant congestion control (CN/CWR flags)
- MTU negotiation (1132-1232 byte range)

## Architecture

Sans-I/O core: the protocol state machine consumes and produces byte buffers without performing actual network I/O. This matches IronRDP's architectural pattern and enables use with any async runtime or blocking I/O.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
