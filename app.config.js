const base = require('./app.json').expo;

module.exports = {
  expo: {
    ...base,
    ios: {
      ...base.ios,
      infoPlist: {
        ...base.ios.infoPlist,
        GEMINI_API_KEY: process.env.GEMINI_API_KEY ?? '',
      },
    },
  },
};
