# Proposed: Subject-framing preflight for i23d (border-cut and off-center subjects)

## Metadata

- Created: 2026-07-20 (webcam-bust redo forensics, operator session)
- Status: Proposed

## Context

The 2026-07-19 webcam demo exposed a framing failure class: a raw 1920x1080
frame with the subject cut at the frame bottom and off-center shredded the
flagship Hunyuan3D-2.1 shape stage (topology_raw 9,592 bodies / euler +17,902;
quality_verdict degraded: "the source pose painted far less than its viewpoint
could see"). The same photo center-cropped to a square bust produced a healthy
watertight mesh through the 2mv multiview lane. TripoSR tolerated the raw
frame (rembg isolation + bust-heavy training) but its unseen angles remain
weak without reference generation.

The failure is preventable BEFORE spending 25+ minutes of shape compute: the
subject mask (already computed by the rembg preprocessing) reveals both
problems — mask touching image borders (truncated silhouette) and a small
subject-to-frame ratio (wasted conditioning resolution).

## Proposal

- After background removal, measure (a) mask-border contact per edge and
  (b) subject bounding-box area ratio.
- Auto-crop to a padded square around the subject mask when the ratio is low
  (general-purpose reframing, not person-specific), recording the crop in
  metadata.
- When the mask touches a border (truncated subject), warn loudly in the
  result notes — a truncated silhouette cannot be un-truncated, and the
  Hunyuan lane should state the risk before the shape stage runs.
- Never silently alter the input beyond the recorded crop; `--no-preflight`
  escape hatch for callers that pre-frame.

## Evidence

- out/laurent-bust-redo/hunyuan_refgen (raw frame: shredded, gated degraded)
- out/laurent-bust-redo/hunyuan_multiview (square crop: healthy, watertight)
- out/laurent-bust-redo/comparison_contact_sheet.png
