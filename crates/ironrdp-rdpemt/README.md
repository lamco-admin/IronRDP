# ironrdp-rdpemt

RDP Multitransport Extension (RDPEMT) tunnel management for [IronRDP](https://github.com/Devolutions/IronRDP).

Implements the multitransport tunnel defined in [MS-RDPEMT](https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-rdpemt/), which negotiates and manages UDP tunnels over an established RDPEUDP2 connection.

## Protocol Coverage

- TLS handshake over RDPEUDP2 (using rustls)
- RDP_TUNNEL_HEADER parsing/encoding with action cycling
- Tunnel Create Request/Response (RequestID + SecurityCookie binding)
- Tunnel Data PDU wrapping/unwrapping for DVC traffic

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
