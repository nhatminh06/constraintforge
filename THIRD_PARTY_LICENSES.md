# Third-party dependency licenses and notices

This candidate distributes project source and research data, not dependency source or compiled binaries. Cargo resolves the public crates listed below from `Cargo.lock`; their project licenses do not relicense those crates. The target columns identify packages in the resolved Apple, Linux, and Windows dependency graphs. The full lockfile contains 172 registry package versions, all with license declarations in their package metadata. No dependency version was upgraded for this review.

The `unicode-ident` declaration is `(MIT OR Apache-2.0) AND Unicode-3.0`: the Unicode terms are additional, not optional. Its exact bundled [`LICENSE-UNICODE`](licenses/third-party/unicode-ident-LICENSE-UNICODE) is retained. The exact [`cfg_aliases` notice](licenses/third-party/cfg_aliases-NOTICES.md) and [`unicode-width` copyright file](licenses/third-party/unicode-width-COPYRIGHT) are also retained. Other crates include MIT, Apache-2.0, Zlib, ISC, BSD-2-Clause, or Unlicense choices as shown below. If a future release distributes compiled binaries or bundles dependency source, that distribution must carry all applicable third-party license texts, copyright notices, and required attribution for the actual included components; this source-only candidate does not qualify such a binary package.

| Crate | Version | Declared license | Apple | Linux | Windows |
|---|---|---|:---:|:---:|:---:|
| aho-corasick | 1.1.5 | Unlicense OR MIT | Yes | Yes | Yes |
| allocator-api2 | 0.2.21 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| android_system_properties | 0.1.6 | MIT OR Apache-2.0 |  |  |  |
| anstream | 1.0.0 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| anstyle | 1.0.14 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| anstyle-parse | 1.0.0 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| anstyle-query | 1.1.5 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| anstyle-wincon | 3.0.11 | MIT OR Apache-2.0 |  |  | Yes |
| arrayvec | 0.7.8 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| ash | 0.38.0+1.3.281 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| autocfg | 1.5.1 | Apache-2.0 OR MIT | Yes | Yes | Yes |
| bit-set | 0.10.0 | Apache-2.0 OR MIT | Yes | Yes | Yes |
| bit-vec | 0.9.1 | Apache-2.0 OR MIT | Yes | Yes | Yes |
| bitflags | 2.13.2 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| block-buffer | 0.10.4 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| block2 | 0.6.2 | MIT | Yes |  |  |
| bumpalo | 3.20.3 | MIT OR Apache-2.0 |  |  |  |
| bytemuck | 1.25.2 | Zlib OR Apache-2.0 OR MIT | Yes | Yes | Yes |
| bytemuck_derive | 1.12.1 | Zlib OR Apache-2.0 OR MIT | Yes | Yes | Yes |
| cfg-if | 1.0.5 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| cfg_aliases | 0.2.2 | MIT | Yes | Yes | Yes |
| clap | 4.6.7 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| clap_builder | 4.6.7 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| clap_derive | 4.6.7 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| clap_lex | 1.1.1 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| codespan-reporting | 0.13.1 | Apache-2.0 | Yes | Yes | Yes |
| colorchoice | 1.0.5 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| cpufeatures | 0.2.17 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| crunchy | 0.2.4 | MIT |  |  |  |
| crypto-common | 0.1.7 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| digest | 0.10.7 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| dispatch2 | 0.3.1 | Zlib OR Apache-2.0 OR MIT | Yes |  |  |
| dlib | 0.5.3 | MIT | Yes | Yes |  |
| document-features | 0.2.12 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| equivalent | 1.0.3 | Apache-2.0 OR MIT | Yes | Yes | Yes |
| foldhash | 0.2.0 | Zlib | Yes | Yes | Yes |
| futures-core | 0.3.34 | MIT OR Apache-2.0 |  |  |  |
| futures-task | 0.3.34 | MIT OR Apache-2.0 |  |  |  |
| futures-util | 0.3.34 | MIT OR Apache-2.0 |  |  |  |
| generic-array | 0.14.7 | MIT | Yes | Yes | Yes |
| gl_generator | 0.14.0 | Apache-2.0 |  |  | Yes |
| glow | 0.17.0 | MIT OR Apache-2.0 OR Zlib | Yes | Yes | Yes |
| glutin_wgl_sys | 0.6.1 | Apache-2.0 |  |  | Yes |
| gpu-allocator | 0.28.0 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| half | 2.7.1 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| hashbrown | 0.16.1 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| hashbrown | 0.17.1 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| heck | 0.5.0 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| indexmap | 2.14.2 | Apache-2.0 OR MIT | Yes | Yes | Yes |
| is_terminal_polyfill | 1.70.2 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| itoa | 1.0.19 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| jni-sys | 0.3.1 | MIT OR Apache-2.0 |  |  |  |
| jni-sys | 0.4.1 | MIT OR Apache-2.0 |  |  |  |
| jni-sys-macros | 0.4.1 | MIT OR Apache-2.0 |  |  |  |
| js-sys | 0.3.107 | MIT OR Apache-2.0 |  |  |  |
| khronos-egl | 6.0.0 | MIT/Apache-2.0 | Yes | Yes | Yes |
| khronos_api | 3.1.0 | Apache-2.0 |  |  | Yes |
| lazy_static | 1.5.1 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| libc | 0.2.191 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| libloading | 0.8.9 | ISC | Yes | Yes | Yes |
| libm | 0.2.16 | MIT | Yes | Yes | Yes |
| litrs | 1.0.1 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| lock_api | 0.4.14 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| log | 0.4.35 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| matchers | 0.2.0 | MIT | Yes | Yes | Yes |
| memchr | 2.8.3 | Unlicense OR MIT | Yes | Yes | Yes |
| naga | 30.0.1 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| naga-types | 30.0.1 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| ndk-sys | 0.6.0+11769913 | MIT OR Apache-2.0 |  |  |  |
| nu-ansi-term | 0.50.3 | MIT | Yes | Yes | Yes |
| num-traits | 0.2.19 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| objc2 | 0.6.5 | MIT | Yes |  |  |
| objc2-core-foundation | 0.3.2 | Zlib OR Apache-2.0 OR MIT | Yes |  |  |
| objc2-core-graphics | 0.3.2 | Zlib OR Apache-2.0 OR MIT | Yes |  |  |
| objc2-encode | 4.1.0 | MIT | Yes |  |  |
| objc2-foundation | 0.3.2 | MIT | Yes |  |  |
| objc2-io-surface | 0.3.2 | Zlib OR Apache-2.0 OR MIT | Yes |  |  |
| objc2-metal | 0.3.2 | Zlib OR Apache-2.0 OR MIT | Yes |  |  |
| objc2-quartz-core | 0.3.2 | Zlib OR Apache-2.0 OR MIT | Yes |  |  |
| once_cell | 1.21.4 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| once_cell_polyfill | 1.70.2 | MIT OR Apache-2.0 |  |  | Yes |
| ordered-float | 5.4.0 | MIT | Yes | Yes | Yes |
| parking_lot | 0.12.5 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| parking_lot_core | 0.9.12 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| pin-project-lite | 0.2.17 | Apache-2.0 OR MIT | Yes | Yes | Yes |
| pkg-config | 0.3.34 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| pollster | 0.4.0 | Apache-2.0/MIT | Yes | Yes | Yes |
| portable-atomic | 1.15.0 | Apache-2.0 OR MIT |  |  |  |
| portable-atomic-util | 0.2.8 | Apache-2.0 OR MIT |  |  |  |
| presser | 0.3.1 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| proc-macro2 | 1.0.107 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| profiling | 1.0.18 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| quote | 1.0.47 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| range-alloc | 0.1.5 | MIT OR Apache-2.0 |  |  | Yes |
| raw-window-handle | 0.6.2 | MIT OR Apache-2.0 OR Zlib | Yes | Yes | Yes |
| raw-window-metal | 1.1.0 | MIT OR Apache-2.0 | Yes |  |  |
| redox_syscall | 0.5.18 | MIT |  |  |  |
| regex-automata | 0.4.18 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| regex-syntax | 0.8.11 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| renderdoc-sys | 1.1.0 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| rustc-hash | 1.1.0 | Apache-2.0/MIT | Yes | Yes | Yes |
| rustversion | 1.0.23 | MIT OR Apache-2.0 |  |  |  |
| scopeguard | 1.2.0 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| serde | 1.0.229 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| serde_core | 1.0.229 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| serde_derive | 1.0.229 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| serde_json | 1.0.154 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| serde_spanned | 1.1.3 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| sha2 | 0.10.9 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| sharded-slab | 0.1.7 | MIT | Yes | Yes | Yes |
| slab | 0.4.12 | MIT |  |  |  |
| slotmap | 1.1.1 | Zlib |  |  |  |
| smallvec | 1.16.5 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| spirv | 0.4.0+sdk-1.4.341.0 | Apache-2.0 | Yes | Yes | Yes |
| static_assertions | 1.1.0 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| strsim | 0.11.1 | MIT | Yes | Yes | Yes |
| syn | 2.0.119 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| syn | 3.0.9 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| termcolor | 1.4.1 | Unlicense OR MIT | Yes | Yes | Yes |
| thiserror | 2.0.21 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| thiserror-impl | 2.0.21 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| thread_local | 1.1.10 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| tokio | 1.54.1 | MIT |  |  |  |
| toml | 0.9.12+spec-1.1.0 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| toml_datetime | 0.7.5+spec-1.1.0 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| toml_parser | 1.1.5+spec-1.1.0 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| toml_writer | 1.2.0+spec-1.1.0 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| tracing | 0.1.44 | MIT | Yes | Yes | Yes |
| tracing-attributes | 0.1.31 | MIT | Yes | Yes | Yes |
| tracing-core | 0.1.36 | MIT | Yes | Yes | Yes |
| tracing-log | 0.2.0 | MIT | Yes | Yes | Yes |
| tracing-subscriber | 0.3.23 | MIT | Yes | Yes | Yes |
| typenum | 1.20.1 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| unicode-ident | 1.0.27 | (MIT OR Apache-2.0) AND Unicode-3.0 | Yes | Yes | Yes |
| unicode-width | 0.2.3 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| utf8parse | 0.2.2 | Apache-2.0 OR MIT | Yes | Yes | Yes |
| valuable | 0.1.1 | MIT |  |  |  |
| version_check | 0.9.5 | MIT/Apache-2.0 | Yes | Yes | Yes |
| wasm-bindgen | 0.2.130 | MIT OR Apache-2.0 |  |  |  |
| wasm-bindgen-futures | 0.4.80 | MIT OR Apache-2.0 |  |  |  |
| wasm-bindgen-macro | 0.2.130 | MIT OR Apache-2.0 |  |  |  |
| wasm-bindgen-macro-support | 0.2.130 | MIT OR Apache-2.0 |  |  |  |
| wasm-bindgen-shared | 0.2.130 | MIT OR Apache-2.0 |  |  |  |
| wayland-sys | 0.31.11 | MIT | Yes | Yes |  |
| web-sys | 0.3.107 | MIT OR Apache-2.0 |  |  |  |
| wgpu | 30.0.1 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| wgpu-core | 30.0.1 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| wgpu-core-deps-apple | 30.0.1 | MIT OR Apache-2.0 | Yes |  |  |
| wgpu-core-deps-emscripten | 30.0.1 | MIT OR Apache-2.0 |  |  |  |
| wgpu-core-deps-windows-linux-android | 30.0.1 | MIT OR Apache-2.0 |  | Yes | Yes |
| wgpu-hal | 30.0.1 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| wgpu-naga-bridge | 30.0.1 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| wgpu-types | 30.0.1 | MIT OR Apache-2.0 | Yes | Yes | Yes |
| winapi-util | 0.1.11 | Unlicense OR MIT |  |  | Yes |
| windows | 0.62.2 | MIT OR Apache-2.0 |  |  | Yes |
| windows-collections | 0.3.2 | MIT OR Apache-2.0 |  |  | Yes |
| windows-core | 0.62.2 | MIT OR Apache-2.0 |  |  | Yes |
| windows-future | 0.3.2 | MIT OR Apache-2.0 |  |  | Yes |
| windows-implement | 0.60.2 | MIT OR Apache-2.0 |  |  | Yes |
| windows-interface | 0.59.3 | MIT OR Apache-2.0 |  |  | Yes |
| windows-link | 0.2.1 | MIT OR Apache-2.0 |  |  | Yes |
| windows-numerics | 0.3.1 | MIT OR Apache-2.0 |  |  | Yes |
| windows-result | 0.4.1 | MIT OR Apache-2.0 |  |  | Yes |
| windows-strings | 0.5.1 | MIT OR Apache-2.0 |  |  | Yes |
| windows-sys | 0.61.2 | MIT OR Apache-2.0 |  |  | Yes |
| windows-threading | 0.2.1 | MIT OR Apache-2.0 |  |  | Yes |
| winnow | 0.7.15 | MIT | Yes | Yes | Yes |
| winnow | 1.0.5 | MIT | Yes | Yes | Yes |
| xml-rs | 0.8.29 | MIT |  |  | Yes |
| zerocopy | 0.8.63 | BSD-2-Clause OR Apache-2.0 OR MIT | Yes | Yes | Yes |
| zerocopy-derive | 0.8.63 | BSD-2-Clause OR Apache-2.0 OR MIT | Yes | Yes | Yes |
| zmij | 1.0.23 | MIT | Yes | Yes | Yes |
