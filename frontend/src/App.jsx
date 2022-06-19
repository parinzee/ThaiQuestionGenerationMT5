import Info from "./components/Info.jsx"
import Settings from "./components/Settings.jsx"

function App() {
  return (
    <div className="grid gap-32 h-screen w-screen grid-cols-1 md:grid-cols-5 bg-neutral-100 dark:bg-neutral-900 min-h-0 min-w-0">
      <div className="grid grid-cols-1 grid-rows-1 col-span-1 md:col-span-3 content-center min-h-0 min-w-0">
        <Info />
      </div>
      <div className="col-span-1 md:col-span-2"></div>
    </div>
  )
}

export default App
