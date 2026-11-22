import React, { useCallback, useEffect, useState } from "react";
import {
  Alert, Box, Button, Chip, IconButton, MenuItem, Paper, Stack, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, TextField, Tooltip, Typography,
} from "@mui/material";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import PageLayout from "@/components/layout/PageLayout";
import { BASE_URL, handleResponse } from "@/api/apiClient";
import { enablePlugin, startPlugin, stopPlugin } from "@/api/pluginsService";

const PLUGIN_ID = "mcbeak";
const STATUS_EVERY_MS = 5000;

const api = (path, options) =>
  fetch(`${BASE_URL}/mcbeak${path}`, options).then(handleResponse).then((d) => d.data || d);
const put = (path, body) =>
  api(path, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

// A Piper voice is one model file with numbered speakers; a Kokoro voice is a name.
const voiceLabel = (row) =>
  row.voice && row.voice.endsWith(".onnx") ? `${row.voice.replace(/\.onnx$/, "")} #${row.voice_speaker}` : row.voice;

const microphone = (status) => {
  if (!status) return { label: "Checking…", color: "default" };
  if (!status.running) return { label: "Not running", color: "default" };
  if (status.listening) return { label: "Microphone open", color: "success" };
  return { label: status.status === "starting" ? "Starting, microphone closed" : "Microphone closed", color: "warning" };
};

const VoiceTeamPage = () => {
  const [status, setStatus] = useState(null);
  const [team, setTeam] = useState(null);
  const [settings, setSettings] = useState(null);
  const [text, setText] = useState("");
  const [loadError, setLoadError] = useState(null);
  const [switchError, setSwitchError] = useState(null);
  const [teamError, setTeamError] = useState(null);
  const [settingsError, setSettingsError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [added, setAdded] = useState({ name: "", kind: "", model: "" });

  const loadStatus = useCallback(() => {
    api("/status").then((s) => { setStatus(s); setLoadError(null); })
      .catch((e) => setLoadError(e.message || "Guaardvark did not answer."));
  }, []);

  useEffect(() => {
    loadStatus();
    api("/team").then(setTeam).catch((e) => setTeamError(e.message));
    api("/settings").then((s) => { setSettings(s); setText(s.text || ""); }).catch((e) => setSettingsError(e.message));
    const timer = setInterval(loadStatus, STATUS_EVERY_MS);
    return () => clearInterval(timer);
  }, [loadStatus]);

  const switchListener = async (on) => {
    setBusy(true);
    setSwitchError(null);
    try {
      if (on) {
        await enablePlugin(PLUGIN_ID);
        await startPlugin(PLUGIN_ID);
      } else {
        await stopPlugin(PLUGIN_ID);
      }
    } catch (e) {
      setSwitchError(e.message || (on ? "The listener did not start." : "The listener did not stop."));
    } finally {
      setBusy(false);
      loadStatus();
    }
  };

  const saveTeam = (members) => {
    setBusy(true);
    return put("/team", { members })
      .then((saved) => { setTeam(saved); setTeamError(null); return true; })
      .catch((e) => { setTeamError(e.message); return false; })
      .finally(() => setBusy(false));
  };

  const members = team ? team.members.map((row) => row.member) : [];
  const setPermissions = (index, value) =>
    saveTeam(members.map((m, i) => (i === index ? { ...m, permissions: value } : m)));
  const remove = (index) => saveTeam(members.filter((_, i) => i !== index));
  const add = async () => {
    const member = { name: added.name.trim(), kind: added.kind };
    if (added.model.trim()) member.model = added.model.trim();
    if (await saveTeam([...members, member])) setAdded({ name: "", kind: "", model: "" });
  };

  const saveSettings = () => {
    setBusy(true);
    put("/settings", { text })
      .then((saved) => { setSettings(saved); setText(saved.text || ""); setSettingsError(null); })
      .catch((e) => setSettingsError(e.message))
      .finally(() => setBusy(false));
  };

  if (loadError) {
    // The backend loads extensions when it starts, so a freshly linked one has no route yet.
    return (
      <PageLayout title="Voice Team">
        <Alert severity="info">
          Guaardvark has not loaded Mr. McBeak yet. Restart Guaardvark, then open this page again. ({loadError})
        </Alert>
      </PageLayout>
    );
  }

  const mic = microphone(status);
  const running = Boolean(status && status.running);

  return (
    <PageLayout title="Voice Team">
      <Stack spacing={3} sx={{ maxWidth: 960 }}>
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" useFlexGap>
            <Chip label={mic.label} color={mic.color} />
            <Button variant="contained" disabled={busy || !status} onClick={() => switchListener(!running)}>
              {running ? "Stop listening" : "Start listening"}
            </Button>
            {running && (
              <Typography variant="body2" color="text.secondary">
                Mr. McBeak {status.version}
                {status.team && status.team.length > 0 ? `, answering to ${status.team.join(", ")}` : ""}
              </Typography>
            )}
          </Stack>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Starting opens this machine's microphone; stopping closes it. A copy started in a terminal without a
            status port shows here as not running.
          </Typography>
          {switchError && <Alert severity="error" sx={{ mt: 1 }}>{switchError}</Alert>}
        </Paper>

        <Box>
          <Typography variant="h6" gutterBottom>Team</Typography>
          {teamError && <Alert severity="error" sx={{ mb: 1 }} onClose={() => setTeamError(null)}>{teamError}</Alert>}
          {team && (
            <>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Name</TableCell>
                      <TableCell>Kind</TableCell>
                      <TableCell>Voice</TableCell>
                      <TableCell>Permissions</TableCell>
                      <TableCell />
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {team.members.map((row, index) => (
                      <TableRow key={row.name}>
                        <TableCell>
                          {row.name}
                          {row.missing_program && (
                            <Chip size="small" color="warning" sx={{ ml: 1 }} label={`${row.missing_program} is not installed`} />
                          )}
                        </TableCell>
                        <TableCell>{row.model ? `${row.kind} (${row.model})` : row.kind}</TableCell>
                        <TableCell>{voiceLabel(row)}</TableCell>
                        <TableCell>
                          <TextField
                            select size="small" variant="standard" value={row.permissions || "readonly"} disabled={busy}
                            onChange={(e) => setPermissions(index, e.target.value)}
                          >
                            <MenuItem value="readonly">readonly</MenuItem>
                            <MenuItem value="full">full</MenuItem>
                          </TextField>
                        </TableCell>
                        <TableCell align="right">
                          <Tooltip title={team.members.length > 1 ? `Remove ${row.name}` : "The team needs at least one member"}>
                            <span>
                              <IconButton size="small" disabled={busy || team.members.length < 2} onClick={() => remove(index)}>
                                <DeleteOutlineIcon fontSize="small" />
                              </IconButton>
                            </span>
                          </Tooltip>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              <Stack direction="row" spacing={1} alignItems="flex-end" flexWrap="wrap" useFlexGap sx={{ mt: 2 }}>
                <TextField
                  size="small" label="Name" helperText="What you will call it out loud" value={added.name}
                  onChange={(e) => setAdded({ ...added, name: e.target.value })}
                />
                <TextField
                  select size="small" label="Kind" helperText=" " sx={{ minWidth: 150 }} value={added.kind}
                  onChange={(e) => setAdded({ ...added, kind: e.target.value })}
                >
                  {team.kinds.map((kind) => <MenuItem key={kind} value={kind}>{kind}</MenuItem>)}
                </TextField>
                <TextField
                  size="small" label="Model" helperText={added.kind === "ollama" ? "Required for ollama" : "Optional"}
                  value={added.model} onChange={(e) => setAdded({ ...added, model: e.target.value })}
                />
                <Box sx={{ pb: "22px" }}>
                  <Button variant="outlined" disabled={busy || !added.name.trim() || !added.kind} onClick={add}>
                    Add member
                  </Button>
                </Box>
              </Stack>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Kept in {team.path}, the file <code>mcbeak team add</code> writes; a running listener picks a change up
                by itself. With full permissions a spoken request may edit files and run commands, and the microphone
                hears everyone in the room. What you say to a {team.cloud_kinds.join(", ")} member goes to its maker's
                servers.
              </Typography>
            </>
          )}
        </Box>

        <Box>
          <Typography variant="h6" gutterBottom>Settings</Typography>
          {settingsError && <Alert severity="error" sx={{ mb: 1 }} onClose={() => setSettingsError(null)}>{settingsError}</Alert>}
          {settings && settings.problem && <Alert severity="warning" sx={{ mb: 1 }}>{settings.problem}</Alert>}
          {settings && (
            <>
              <TextField
                multiline fullWidth minRows={8} value={text} onChange={(e) => setText(e.target.value)}
                placeholder={"emotion:\n  enabled: true"}
                InputProps={{ sx: { fontFamily: "monospace", fontSize: 13 } }}
              />
              <Stack direction="row" spacing={2} alignItems="center" sx={{ mt: 1 }}>
                <Button variant="contained" disabled={busy || text === (settings.text || "")} onClick={saveSettings}>
                  Save settings
                </Button>
                <Typography variant="body2" color="text.secondary">
                  {settings.path} is laid over Mr. McBeak's config.yaml key by key. A change applies the next time the
                  listener starts.
                </Typography>
              </Stack>
            </>
          )}
        </Box>
      </Stack>
    </PageLayout>
  );
};

export default VoiceTeamPage;
