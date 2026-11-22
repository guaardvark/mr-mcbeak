// The Voice Team page in Guaardvark's web UI. See frontend/src/extensions.js there for the contract.
import React, { lazy } from "react";
import RecordVoiceOverIcon from "@mui/icons-material/RecordVoiceOver";

const VoiceTeamPage = lazy(() => import("./VoiceTeamPage"));

export default {
  routes: [{ path: "/voice-team", element: <VoiceTeamPage /> }],
  navGroups: [{ label: "Voice", items: [{ text: "Voice Team", icon: <RecordVoiceOverIcon />, path: "/voice-team" }] }],
  pageContext: { routeMap: { "/voice-team": { page: "Voice Team", entityType: null } } },
};
