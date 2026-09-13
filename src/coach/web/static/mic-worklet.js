/**
 * Microphone capture worklet.
 *
 * Browsers hand you audio at the hardware rate — usually 48 kHz, sometimes 44.1 kHz — in
 * 128-sample blocks of float32. The speech recogniser wants 16 kHz mono PCM16. Resampling
 * here rather than on the server cuts what goes over the socket by two thirds and means the
 * server never has to care what audio hardware the user has.
 *
 * Runs on the audio thread, so it must stay cheap: linear interpolation, no allocation in the
 * steady state beyond the outgoing frame.
 */

const TARGET_RATE = 16000;
const FRAME_MS = 100;
const FRAME_SAMPLES = (TARGET_RATE * FRAME_MS) / 1000; // 1600

class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE; // `sampleRate` is a worklet global
    this.pos = 0;          // fractional read position into the incoming stream
    this.tail = new Float32Array(0); // samples carried over between blocks
    this.out = new Int16Array(FRAME_SAMPLES);
    this.outLen = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    // Join the carry-over with this block so interpolation can span the boundary.
    const block = input[0];
    const buf = new Float32Array(this.tail.length + block.length);
    buf.set(this.tail, 0);
    buf.set(block, this.tail.length);

    let p = this.pos;
    while (p < buf.length - 1) {
      const i = Math.floor(p);
      const frac = p - i;
      const sample = buf[i] * (1 - frac) + buf[i + 1] * frac;
      // Clamp before scaling: values outside [-1, 1] would wrap and sound like clicks.
      const clamped = Math.max(-1, Math.min(1, sample));
      this.out[this.outLen++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;

      if (this.outLen === FRAME_SAMPLES) {
        // Transfer rather than copy — this is the audio thread.
        const frame = this.out.slice();
        this.port.postMessage(frame.buffer, [frame.buffer]);
        this.outLen = 0;
      }
      p += this.ratio;
    }

    // Keep the unconsumed tail and rebase the read position onto it.
    const consumed = Math.floor(p);
    this.tail = buf.slice(consumed);
    this.pos = p - consumed;
    return true;
  }
}

registerProcessor("mic-processor", MicProcessor);
